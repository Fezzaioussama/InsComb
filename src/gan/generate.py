#!/usr/bin/env python3
# generates synthetic images with a generator trained by train_gan.py, to add them to the training set
#
# only add them to the training split: the validation and test sets must stay real photos
#
# usage: python -m src.gan.generate models/gan/Propre/gan_checkpoint.pth.tar data/bottles/train/Propre --count 1000 --size 299
import argparse
import os

import torch
from PIL import Image

from src.gan.dcgan import Generator

# parse command line
parser = argparse.ArgumentParser(description='Generate synthetic images with a trained DCGAN generator')
parser.add_argument('checkpoint', help='GAN checkpoint saved by train_gan.py (gan_checkpoint.pth.tar)')
parser.add_argument('output_dir', help='folder to write the images to, e.g. data/bottles/train/Propre')
parser.add_argument('--count', default=500, type=int, help='number of images to generate (default: 500)')
parser.add_argument('--size', default=None, type=int, help='resize the images to NxN like the dataset, e.g. 224 or 299 (default: GAN image size)')
parser.add_argument('--prefix', default='gan', help='file name prefix of the images: <prefix>_00000.jpg, ... (default: gan)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='number of images generated at once (default: 64)')
parser.add_argument('--seed', default=None, type=int, help='random seed, to generate the same images again')
args = parser.parse_args()

if {'val', 'test'} & set(os.path.normpath(os.path.abspath(args.output_dir)).split(os.sep)):
    print("=> WARNING: generated images should only be added to the training set, the val/test sets must stay real photos")

if args.seed is not None:
    torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# rebuild the generator with the sizes saved in the checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)
generator = Generator(checkpoint['nz'], checkpoint['ngf'], checkpoint['image_size']).to(device)
generator.load_state_dict(checkpoint['generator'])
generator.eval()
print(f"=> loaded generator of {checkpoint['image_size']}x{checkpoint['image_size']} images (epoch {checkpoint['epoch']})")

os.makedirs(args.output_dir, exist_ok=True)
generated = 0

with torch.no_grad():
    while generated < args.count:
        batch = min(args.batch_size, args.count - generated)
        images = generator(torch.randn(batch, checkpoint['nz'], 1, 1, device=device))

        # from [-1, 1] floats to 0-255 pixels, N x H x W x C
        images = ((images + 1) * 127.5).clamp(0, 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

        for image in images:
            image = Image.fromarray(image)
            if args.size:
                image = image.resize((args.size, args.size), Image.BICUBIC)

            image.save(os.path.join(args.output_dir, f'{args.prefix}_{generated:05d}.jpg'), quality=95)
            generated += 1

print(f"done: {generated} images written to {args.output_dir}")
