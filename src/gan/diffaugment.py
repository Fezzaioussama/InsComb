"""
Differentiable augmentation for data-efficient GAN training (DiffAugment, Zhao et al., 2020).

The same random augmentations are applied to the real and the generated images before the
discriminator, so that it cannot memorize the few training photos. They are differentiable,
so the generator is still trained through them, and they do not appear in the generated images
"""
import torch
import torch.nn.functional as F


def color(images):
    """
    Random brightness, saturation and contrast of each image (in [-1, 1])
    """
    def factor():
        return torch.rand(images.size(0), 1, 1, 1, device=images.device)

    images = images + (factor() - 0.5)
    mean = images.mean(dim=1, keepdim=True)
    images = (images - mean) * (factor() * 2) + mean
    mean = images.mean(dim=[1, 2, 3], keepdim=True)
    return (images - mean) * (factor() + 0.5) + mean


def translation(images, ratio=0.125):
    """
    Shift each image by up to 1/8 of its size, filling the border with zeros
    """
    n, _, height, width = images.shape
    pad = int(height * ratio + 0.5)
    padded = F.pad(images, [pad, pad, pad, pad])

    dy = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    dx = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    return torch.stack([padded[i, :, dy[i]:dy[i] + height, dx[i]:dx[i] + width] for i in range(n)])


def cutout(images, ratio=0.5):
    """
    Zero a random square of half the image size in each image
    """
    n, _, height, width = images.shape
    size = int(height * ratio + 0.5)

    mask = torch.ones(n, 1, height, width, device=images.device)
    ys = torch.randint(0, height - size + 1, (n,)).tolist()
    xs = torch.randint(0, width - size + 1, (n,)).tolist()
    for i in range(n):
        mask[i, :, ys[i]:ys[i] + size, xs[i]:xs[i] + size] = 0

    return images * mask


AUGMENTATIONS = {
    'color': color,
    'translation': translation,
    'cutout': cutout,
}


def diff_augment(images, policy):
    """
    Apply the augmentations of the policy, e.g. ['translation', 'cutout']
    """
    for name in policy:
        images = AUGMENTATIONS[name](images)

    return images
