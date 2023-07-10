#!/usr/bin/env python3
# runs an exported ONNX model over a folder of images and counts the predictions per class
import argparse
import os
import torch
from PIL import Image
import onnxruntime
import torchvision.transforms as transforms

# parse command line
parser = argparse.ArgumentParser(description='Evaluate an exported ONNX model on a folder of images')
parser.add_argument('model', help='path to the ONNX model')
parser.add_argument('folder', help='folder of test images (e.g. data/test/Propre)')
parser.add_argument('--resolution', default=299, type=int, help='input NxN resolution of the model (default: 299 for Inception)')
parser.add_argument('--classes', default='Defective,Propre', help='comma-separated class names in model output order (default: Defective,Propre)')
args = parser.parse_args()

# Load the ONNX model
pytorch_model = onnxruntime.InferenceSession(args.model)
input_name = pytorch_model.get_inputs()[0].name
output_name = pytorch_model.get_outputs()[0].name

# Define the image preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(args.resolution),
    transforms.CenterCrop(args.resolution),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the list of class names
class_names = args.classes.split(',')
counts = {name: 0 for name in class_names}

# Loop through the images in the folder
for filename in sorted(os.listdir(args.folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Open the image
        img_path = os.path.join(args.folder, filename)
        photo = Image.open(img_path).convert('RGB')

        # Preprocess the image and perform inference
        with torch.no_grad():
            photo_tensor = preprocess(photo).unsqueeze(0)
            output = pytorch_model.run([], {input_name: photo_tensor.numpy()})[0]

        # Get the index of the highest probability for the output
        pred = output.argmax()

        # Look up the corresponding class name
        class_name = class_names[pred]
        counts[class_name] += 1

        # Print the result
        print(filename, class_name, 'acc=', max(output[0]))

for name, count in counts.items():
    print(f'{name}: {count}')
