import argparse
import os

import torch

from core.compiler import fusedinf_compiler
from models import *
from models import ResNet18, vgg16

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--input_dir', type=str, default='inputs')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    model_dir = args.model_dir
    input_dir = args.input_dir

    files = os.listdir(model_dir)
    input_files = os.listdir(input_dir)

    models = []
    inputs = []

    for file in files:
        if file.endswith('.py'):
            exec_str = f'{file[:-3]}.{file[:-3]}()'
            exec(f'net = {exec_str}')
            models.append(net)

    for file in input_files:
        if file.endswith('.pth'):
            input = torch.load(f'{input_dir}/{file}')
            input.to(device=args.device)
            inputs.append(inputs)

    compiled_model = fusedinf_compiler(models)
    compiled_model.eval()
    compiled_model.to(args.device)
    print(compiled_model(inputs))
