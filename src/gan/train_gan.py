#!/usr/bin/env python3
# trains a DCGAN on the photos of one class, to generate synthetic training images with generate.py
#
# only train it on the training split (e.g. data/bottles/train/Propre): a GAN that has seen the
# validation or test photos would leak them into the training set through its generated images
#
# usage: python -m src.gan.train_gan data/bottles/train/Propre --model-dir models/gan/Propre
import argparse
import datetime
import os
import random
import time

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from src.gan.dcgan import Discriminator, Generator
from src.gan.diffaugment import AUGMENTATIONS, diff_augment

# parse command line
parser = argparse.ArgumentParser(description='Train a DCGAN on the images of one class')
parser.add_argument('data', help='folder with the training images of one class, e.g. data/bottles/train/Propre')
parser.add_argument('--model-dir', default='models/gan', help='output directory for the checkpoint, sample images and tensorboard logs (default: models/gan)')
parser.add_argument('--image-size', default=64, type=int, help='size of the generated images, a power of 2 (default: 64)')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run (default: 300)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate of both networks (default: 2e-4)')
parser.add_argument('--nz', default=100, type=int, help='size of the random latent vector (default: 100)')
parser.add_argument('--ngf', default=64, type=int, help='number of feature maps of the generator (default: 64)')
parser.add_argument('--ndf', default=64, type=int, help='number of feature maps of the discriminator (default: 64)')
parser.add_argument('--diffaugment', nargs='*', choices=list(AUGMENTATIONS), default=['translation', 'cutout'], metavar='NAME',
                    help='DiffAugment augmentations applied to the real and generated images before the discriminator, among: '
                         f"{' '.join(AUGMENTATIONS)} (default: translation cutout, no name to disable)")
parser.add_argument('--include-augmented', action='store_true', help='also train on the images of data_augmentation.py (*_aug_*), skipped by default')
parser.add_argument('--sample-every', default=10, type=int, help='save a grid of generated images every N epochs (default: 10)')
parser.add_argument('--resume', default='', type=str, help='path to a checkpoint to resume training from')
parser.add_argument('-j', '--workers', default=2, type=int, help='number of data loading workers (default: 2)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')


class ImageDataset(torch.utils.data.Dataset):
    """
    Images of one folder, without the augmented copies (unless include_augmented) and the GAN images
    """
    def __init__(self, folder, transform, include_augmented=False):
        self.paths = [os.path.join(folder, filename) for filename in sorted(os.listdir(folder))
                      if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
                      and not filename.startswith('gan_')
                      and (include_augmented or '_aug_' not in filename)]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert('RGB'))


def save_samples(generator, noise, path, tensorboard, epoch):
    """
    Save a grid of images generated from the same noise at every call, to follow the progress
    """
    generator.eval()
    with torch.no_grad():
        images = generator(noise)
    generator.train()

    grid = vutils.make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, path)
    tensorboard.add_image('samples', grid, epoch)
    print(f"saved generated samples to:  {path}")


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # train on the GPU when available, on the CPU otherwise
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=> training on {device}")

    # resuming keeps the network sizes of the checkpoint
    checkpoint = None
    if args.resume:
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        args.image_size, args.nz, args.ngf, args.ndf = (checkpoint['image_size'], checkpoint['nz'],
                                                        checkpoint['ngf'], checkpoint['ndf'])

    # resize the whole bottle to a square, like the dataset images, and scale to [-1, 1] like the generator output
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageDataset(args.data, transform, args.include_augmented)
    if len(dataset) == 0:
        parser.error(f"no training images found in {args.data}")
    print(f"=> {len(dataset)} training images in {args.data}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, pin_memory=device.type == 'cuda')

    # create the networks, with the Adam settings of the DCGAN paper
    generator = Generator(args.nz, args.ngf, args.image_size).to(device)
    discriminator = Discriminator(args.ndf, args.image_size).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if checkpoint:
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"=> resuming from epoch {start_epoch}")

    # output directories and tensorboard logger (to model_dir/tensorboard)
    samples_dir = os.path.join(args.model_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    tensorboard = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    print(f"To start tensorboard run:  tensorboard --logdir={os.path.join(args.model_dir, 'tensorboard')}")

    # same noise for all the sample grids
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        loss_d_sum = loss_g_sum = real_score_sum = fake_score_sum = 0.0

        for real in loader:
            real = real.to(device, non_blocking=True)
            noise = torch.randn(real.size(0), args.nz, 1, 1, device=device)
            fake = generator(noise)

            # (1) update the discriminator: real images -> 0.9 (one-sided label smoothing), generated -> 0
            real_logits = discriminator(diff_augment(real, args.diffaugment))
            fake_logits = discriminator(diff_augment(fake.detach(), args.diffaugment))
            loss_d = criterion(real_logits, torch.full_like(real_logits, 0.9)) + \
                     criterion(fake_logits, torch.zeros_like(fake_logits))

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # (2) update the generator: make the discriminator score the generated images as real
            fake_logits = discriminator(diff_augment(fake, args.diffaugment))
            loss_g = criterion(fake_logits, torch.ones_like(fake_logits))

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # record the losses and the mean discriminator scores D(x) and D(G(z))
            loss_d_sum += loss_d.item()
            loss_g_sum += loss_g.item()
            real_score_sum += torch.sigmoid(real_logits).mean().item()
            fake_score_sum += torch.sigmoid(fake_logits).mean().item()

        batches = len(loader)
        print(f"Epoch: [{epoch}] Loss_D {loss_d_sum / batches:.4f}  Loss_G {loss_g_sum / batches:.4f}  "
              f"D(x) {real_score_sum / batches:.3f}  D(G(z)) {fake_score_sum / batches:.3f}  "
              f"({time.time() - epoch_start:.1f} seconds)")

        tensorboard.add_scalar('Loss/discriminator', loss_d_sum / batches, epoch)
        tensorboard.add_scalar('Loss/generator', loss_g_sum / batches, epoch)
        tensorboard.add_scalar('Score/real', real_score_sum / batches, epoch)
        tensorboard.add_scalar('Score/generated', fake_score_sum / batches, epoch)

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            save_samples(generator, fixed_noise, os.path.join(samples_dir, f'epoch_{epoch:04d}.png'), tensorboard, epoch)

        # save the checkpoint (generate.py rebuilds the generator from it)
        torch.save({
            'epoch': epoch,
            'image_size': args.image_size,
            'nz': args.nz,
            'ngf': args.ngf,
            'ndf': args.ndf,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }, os.path.join(args.model_dir, 'gan_checkpoint.pth.tar'))

    print(f"saved GAN checkpoint to:  {os.path.join(args.model_dir, 'gan_checkpoint.pth.tar')}")


if __name__ == '__main__':
    main(parser.parse_args())
