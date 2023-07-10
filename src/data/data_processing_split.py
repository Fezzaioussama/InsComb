import argparse
import os
import cv2
import numpy as np
import random

# parse command-line arguments
parser = argparse.ArgumentParser(description='Resize the images of one class and split them into train/val/test')
parser.add_argument('input_dir', help='folder containing the (augmented) images of one class')
parser.add_argument('output_dir', help='dataset root, images are written to <output_dir>/{train,val,test}/<class-name>')
parser.add_argument('--class-name', required=True, help='class folder name, e.g. Propre or Defective')
parser.add_argument('--size', default=224, type=int, help='target NxN image size (default: 224, use 299 for Inception)')
parser.add_argument('--train-ratio', default=0.75, type=float, help='fraction of images used for training (default: 0.75)')
parser.add_argument('--val-ratio', default=0.15, type=float, help='fraction of images used for validation (default: 0.15)')
args = parser.parse_args()

# Set the directory paths
dir_path = args.input_dir
train_path = os.path.join(args.output_dir, 'train', args.class_name)
valid_path = os.path.join(args.output_dir, 'val', args.class_name)
test_path = os.path.join(args.output_dir, 'test', args.class_name)

# Set the target size for resizing the images
target_size = (args.size, args.size)

# Set the ratios for splitting the data
train_ratio = args.train_ratio
valid_ratio = args.val_ratio

# Create empty lists to hold the image filenames and labels
image_filenames = []
labels = []

# Loop through the images in the directory
for filename in os.listdir(dir_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_filenames.append(os.path.join(dir_path, filename))
        labels.append(args.class_name.lower())

# Shuffle the image filenames and labels together
combined = list(zip(image_filenames, labels))
random.shuffle(combined)
image_filenames, labels = zip(*combined)

# Compute the number of images for each split
num_train = int(len(image_filenames) * train_ratio)
num_valid = int(len(image_filenames) * valid_ratio)
num_test = len(image_filenames) - num_train - num_valid

# Create the output directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Load, resize, normalize, and save images to respective directories
for i, (filename, label) in enumerate(zip(image_filenames, labels)):
    img = cv2.imread(filename)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.

    if i < num_train:
        cv2.imwrite(os.path.join(train_path, f'{label}_{i}.jpg'), img*255)
    elif i < num_train + num_valid:
        cv2.imwrite(os.path.join(valid_path, f'{label}_{i}.jpg'), img*255)
    else:
        cv2.imwrite(os.path.join(test_path, f'{label}_{i}.jpg'), img*255)

    if i % 100 == 0:
        print(f'{i} images processed...')

print(f'done: {num_train} train, {num_valid} val, {num_test} test images')
