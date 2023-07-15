"""
Resize the images of one class and split them into train / val / test folders, keeping all the
augmented versions of a photo in the same folder

usage: python -m src.data.data_processing_split augmented/Propre data/bottles --class-name Propre --size 299
"""
import argparse
import os
import random
import cv2

# parse command-line arguments
parser = argparse.ArgumentParser(description='Resize the images of one class and split them into train/val/test')
parser.add_argument('input_dir', help='folder containing the (augmented) images of one class')
parser.add_argument('output_dir', help='dataset root, images are written to <output_dir>/{train,val,test}/<class-name>')
parser.add_argument('--class-name', required=True, help='class folder name, e.g. Propre or Defective')
parser.add_argument('--size', default=224, type=int, help='target NxN image size (default: 224, use 299 for Inception)')
parser.add_argument('--train-ratio', default=0.75, type=float, help='fraction of the original photos used for training (default: 0.75)')
parser.add_argument('--val-ratio', default=0.15, type=float, help='fraction of the original photos used for validation (default: 0.15)')
parser.add_argument('--seed', default=42, type=int, help='random seed of the split, to get the same split again (default: 42)')
args = parser.parse_args()

# Set the directory paths
dir_path = args.input_dir
split_paths = {
    'train': os.path.join(args.output_dir, 'train', args.class_name),
    'val': os.path.join(args.output_dir, 'val', args.class_name),
    'test': os.path.join(args.output_dir, 'test', args.class_name),
}

# Set the target size for resizing the images
target_size = (args.size, args.size)


def original_name(filename):
    """
    Name of the original photo of an image: data_augmentation.py names its outputs <original>_aug_<augmentation>.jpeg
    """
    return os.path.splitext(filename)[0].rsplit('_aug_', 1)[0]


# Group the images by original photo, so that all the augmented versions of a photo end up in the
# same split: otherwise near-copies of the training images leak into the validation and test sets
groups = {}
for filename in sorted(os.listdir(dir_path)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        groups.setdefault(original_name(filename), []).append(filename)

if not groups:
    parser.error(f'no images found in {dir_path}')

# Shuffle the original photos and split them with the given ratios
originals = sorted(groups)
random.Random(args.seed).shuffle(originals)

num_train = int(len(originals) * args.train_ratio)
num_valid = int(len(originals) * args.val_ratio)

splits = {
    'train': originals[:num_train],
    'val': originals[num_train:num_train + num_valid],
    'test': originals[num_train + num_valid:],
}

# Load, resize and save the images to their split directory
processed = 0
for split, split_originals in splits.items():
    os.makedirs(split_paths[split], exist_ok=True)
    num_images = 0

    for original in split_originals:
        for filename in groups[original]:
            img = cv2.imread(os.path.join(dir_path, filename))
            if img is None:
                print(f'skipping unreadable image {filename}')
                continue

            img = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(split_paths[split], f'{os.path.splitext(filename)[0]}.jpg'), img)
            num_images += 1

            processed += 1
            if processed % 100 == 0:
                print(f'{processed} images processed...')

    print(f'{split}: {num_images} images from {len(split_originals)} original photos -> {split_paths[split]}')
