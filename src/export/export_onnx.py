#!/usr/bin/env python3
# converts a saved PyTorch model to ONNX format
#
# usage: python -m src.export.export_onnx --model-dir models/inception_v3
import os
import argparse
import inspect

import numpy as np
import onnx
import onnxruntime
import torch

from src.training.reshape import create_model


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth.tar', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")
parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")
parser.add_argument('--no-activation', action='store_true', help="disable adding a Softmax layer to the model (default is to add it)")
parser.add_argument('--opset', type=int, default=13, help="ONNX opset version (default: 13, supported by TensorRT 8.x on the Jetson Nano)")

args = parser.parse_args()
print(args)

# format input model path
if args.model_dir:
    args.model_dir = os.path.expanduser(args.model_dir)
    args.input = os.path.join(args.model_dir, args.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('=> running on device ' + str(device))

# load the model checkpoint (map_location allows loading a GPU checkpoint on a CPU-only machine)
print('=> loading checkpoint:  ' + args.input)
checkpoint = torch.load(args.input, map_location=device)
arch = checkpoint['arch']

# create the model architecture, exactly like train.py did (checkpoints from before
# the 'pretrained' and 'dropout' options were always pre-trained without dropout)
print('=> using model:  ' + arch)
model = create_model(arch, checkpoint['num_classes'],
                     pretrained=checkpoint.get('pretrained', True),
                     dropout=checkpoint.get('dropout', 0.0))

# load the model weights
model.load_state_dict(checkpoint['state_dict'])

# add softmax layer
if not args.no_activation:
    print('=> adding nn.Softmax layer to model')
    model = torch.nn.Sequential(model, torch.nn.Softmax(1))

model.to(device)
model.eval()

# create example image data
resolution = checkpoint['resolution']
input = torch.ones((1, 3, resolution, resolution)).to(device)
print('=> input size:  {:d}x{:d}'.format(resolution, resolution))

# format output model path
if not args.output:
    args.output = arch + '.onnx'

if args.model_dir and args.output.find('/') == -1 and args.output.find('\\') == -1:
    args.output = os.path.join(args.model_dir, args.output)

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

# the Jetson Nano (JetPack 4.x) needs a single-file model with an older opset, so use the
# TorchScript-based exporter: the torch.export-based one, default since PyTorch 2.9,
# needs opset >= 18 and saves the weights in a separate .onnx.data file
export_options = {'opset_version': args.opset}

if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
    export_options['dynamo'] = False

print('=> exporting model to ONNX (opset {:d})...'.format(args.opset))
torch.onnx.export(model, (input,), args.output, input_names=input_names, output_names=output_names, **export_options)
print('=> model exported to:  {:s}'.format(args.output))

# save the class names next to the ONNX model, where the inference scripts look for them
if 'classes' in checkpoint:
    labels_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'labels.txt')
    with open(labels_path, 'w') as file:
        for label in checkpoint['classes']:
            file.write(f"{label}\n")
    print('=> class labels saved to:  {:s}'.format(labels_path))

# check the exported model and compare its output with PyTorch on a random image
onnx.checker.check_model(args.output)
test_input = torch.randn((1, 3, resolution, resolution)).to(device)

with torch.no_grad():
    expected = model(test_input).cpu().numpy()

session = onnxruntime.InferenceSession(args.output, providers=['CPUExecutionProvider'])
actual = session.run(None, {input_names[0]: test_input.cpu().numpy()})[0]

difference = np.abs(expected - actual).max()
print('=> max difference between the PyTorch and ONNX Runtime outputs:  {:.2e}'.format(difference))

if difference > 1e-3:
    print('=> WARNING: the ONNX model output differs from the PyTorch model')
