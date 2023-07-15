#!/usr/bin/env python3
# evaluates an exported ONNX model on a test set: accuracy, precision / recall / F1 per class,
# confusion matrix and inference time
#
# usage: python -m src.export.evaluate_onnx models/inception_v3/inception_v3.onnx data/bottles/test --show-errors
import argparse
import os
import time

import numpy as np
from PIL import Image

from src.inference.onnx_classifier import OnnxClassifier

# parse command line
parser = argparse.ArgumentParser(description='Evaluate an exported ONNX model on a test set')
parser.add_argument('model', help='path to the ONNX model')
parser.add_argument('data', help='test set folder with one subfolder per class (e.g. data/bottles/test)')
parser.add_argument('--labels', default=None, help='class names file (default: labels.txt next to the model)')
parser.add_argument('--show-errors', action='store_true', help='list the misclassified images')
args = parser.parse_args()

# Load the ONNX model
classifier = OnnxClassifier(args.model, args.labels)
classes = classifier.classes
print(f"=> running on {classifier.provider} at {classifier.resolution}x{classifier.resolution}, classes {classes}")

# list the test images: one subfolder per class, like the train/ and val/ folders
samples = []
for true_class in sorted(os.listdir(args.data)):
    class_dir = os.path.join(args.data, true_class)
    if os.path.isdir(class_dir):
        samples += [(os.path.join(class_dir, filename), true_class) for filename in sorted(os.listdir(class_dir))
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not samples:
    parser.error(f"no images found in the class subfolders of {args.data} (e.g. data/bottles/test/Propre/*.jpg)")

# the class folders of the test set must be classes of the model
unknown = sorted({true_class for _, true_class in samples} - set(classes))
if unknown:
    parser.error(f"test set classes {unknown} are not classes of the model {classes}")

# confusion matrix: rows are the true classes, columns the predicted classes
confusion = np.zeros((len(classes), len(classes)), dtype=int)
errors = []
elapsed = 0.0

for path, true_class in samples:
    image = Image.open(path).convert('RGB')

    start = time.time()
    predicted, probability = classifier(image)
    elapsed += time.time() - start

    confusion[classes.index(true_class), classes.index(predicted)] += 1

    if predicted != true_class:
        errors.append((path, true_class, predicted, probability))

total = confusion.sum()
print(f"=> {total} images, {1000 * elapsed / total:.1f} ms per image (preprocessing + inference)")
print(f"=> accuracy:  {100 * np.trace(confusion) / total:.2f}%")
print()

# precision / recall / F1 of each class
print(f"{'class':<12}{'precision':>11}{'recall':>11}{'F1':>11}{'images':>9}")
for i, name in enumerate(classes):
    precision = confusion[i, i] / max(confusion[:, i].sum(), 1)
    recall = confusion[i, i] / max(confusion[i, :].sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(f"{name:<12}{100 * precision:>10.2f}%{100 * recall:>10.2f}%{100 * f1:>10.2f}%{confusion[i].sum():>9}")
print()

print("confusion matrix (rows: true class, columns: predicted class)")
width = max(len(name) for name in classes) + 2
print(' ' * width + ''.join(f"{name:>{width}}" for name in classes))
for i, name in enumerate(classes):
    print(f"{name:<{width}}" + ''.join(f"{count:>{width}}" for count in confusion[i]))

if args.show_errors:
    print()
    print(f"misclassified images ({len(errors)}):")
    for path, true_class, predicted, probability in errors:
        print(f"  {path}  true={true_class}  predicted={predicted} ({probability:.2f})")
