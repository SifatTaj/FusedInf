import argparse
import os

from core.compiler import fusedinf_compiler
from models import *
from models import ResNet18, vgg16

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--input_dir', type=str, default='inputs')

    args = parser.parse_args()
    model_dir = args.model_dir

    files = os.listdir(model_dir)

    models = []

    for file in files:
        if file.endswith('.py'):
            exec_str = f'{file[:-3]}.{file[:-3]}()'
            exec(f'net = {exec_str}')
            models.append(net)

    compiled_model = fusedinf_compiler(models)