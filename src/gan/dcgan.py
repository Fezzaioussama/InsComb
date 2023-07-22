"""
DCGAN generator and discriminator (Radford et al., 2015) used to generate synthetic bottle images.
The discriminator uses spectral normalization for a more stable training on a small dataset, and a
minibatch standard deviation layer against mode collapse (the generator producing the same image)
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def num_blocks(image_size):
    """
    Number of intermediate up/down-sampling blocks between 4x4 and the image size (a power of 2, at least 32)
    """
    blocks = int(math.log2(image_size)) - 3

    if blocks < 2 or image_size != 2 ** (blocks + 3):
        raise ValueError(f"the image size must be a power of 2 of at least 32, got {image_size}")

    return blocks


def init_weights(module):
    """
    DCGAN initialization: convolution weights from N(0, 0.02), batch norm scales from N(1, 0.02)
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class MinibatchStdDev(nn.Module):
    """
    Append the standard deviation of the features across the batch as an extra feature map (Karras et al., 2018),
    so that the discriminator can tell a batch of too similar generated images from a varied batch of real ones
    """
    def forward(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8).mean()
        return torch.cat([x, std.expand(x.size(0), 1, x.size(2), x.size(3))], dim=1)


class Generator(nn.Module):
    """
    Turn a random latent vector (nz x 1 x 1) into an RGB image in [-1, 1], doubling its size with transposed convolutions
    """
    def __init__(self, nz=100, ngf=64, image_size=64):
        super().__init__()
        blocks = num_blocks(image_size)
        mult = 2 ** blocks

        # nz x 1 x 1 -> (ngf * mult) x 4 x 4
        layers = [nn.ConvTranspose2d(nz, ngf * mult, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(ngf * mult),
                  nn.ReLU(True)]

        # double the size and halve the feature maps until image_size / 2
        for _ in range(blocks):
            layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ngf * mult // 2),
                       nn.ReLU(True)]
            mult //= 2

        # ngf x image_size / 2 -> 3 x image_size, in [-1, 1]
        layers += [nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                   nn.Tanh()]

        self.main = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    Score images as real (high) or generated (low), one logit per image
    """
    def __init__(self, ndf=64, image_size=64):
        super().__init__()
        blocks = num_blocks(image_size)

        # 3 x image_size -> ndf x image_size / 2
        layers = [spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1)),
                  nn.LeakyReLU(0.2, inplace=True)]

        # halve the size and double the feature maps until 4x4
        mult = 1
        for _ in range(blocks):
            layers += [spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2, 1)),
                       nn.LeakyReLU(0.2, inplace=True)]
            mult *= 2

        # (ndf * mult + 1 batch std map) x 4 x 4 -> 1 x 1 x 1
        layers += [MinibatchStdDev(),
                   spectral_norm(nn.Conv2d(ndf * mult + 1, 1, 4, 1, 0))]

        self.main = nn.Sequential(*layers)

    def forward(self, images):
        return self.main(images).view(-1)
