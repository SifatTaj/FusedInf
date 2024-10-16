import copy
import time

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.fx.experimental.optimization import replace_node_module

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def fuse_ops(fx_model):
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op != 'call_module':  # If our current node isn't calling a Module then we can ignore it.
            continue

        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue

            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)

    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model

class UnifiedModel(torch.nn.Module):
    def __init__(self, models):
        super(UnifiedModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = []
        for i, net in enumerate(self.models):
            outputs.append(net(x[i]))
        return outputs

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def fusedinf_compiler(models):

    unified_model = UnifiedModel(models)
    print('total params:', count_params(unified_model))
    unified_model = torch.fx.symbolic_trace(unified_model).eval()
    unified_model = fuse_ops(unified_model)
    print('total params:', count_params(unified_model))

    return unified_model