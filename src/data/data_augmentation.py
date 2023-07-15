"""
Generate augmented copies of the photos of one class, to enlarge and diversify the dataset

usage: python -m src.data.data_augmentation raw/Propre augmented/Propre --include-original
       (add --augmentations rotate stretch contrast blur cutout noise for the augmentations of the report)
"""
import argparse
import cv2
import os
import shutil
import numpy as np
import torch
from torchvision.transforms import ColorJitter
from PIL import Image

# define augmentation functions
def luminosity(image):
    """Darken the image, as with weaker lighting"""
    # randomly darken the image by a factor between 0.6 and 0.8
    alpha = np.random.uniform(0.6, 0.8)
    # apply the value to the image
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def color_filter(image):
    """Randomly change the brightness, contrast, saturation and hue of the image"""
    brightness = np.random.uniform(0.6, 0.8)
    contrast = np.random.uniform(0.4, 1.2)
    saturation = np.random.uniform(0.4, 1.1)
    hue = abs(np.random.uniform(-0.3, 0.3))
    transform = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    image = Image.fromarray(image)
    return np.array(transform(image))

def rotate(image):
    """Rotate the image around its center by a random angle (-80 to 80 degrees), to simulate other orientations"""
    angle = np.random.randint(-80, 80)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))

def stretch(image):
    """Stretch the image vertically (about 3 times), to simulate size and perspective variations"""
    scale = np.random.uniform(3, 3.1)
    height, width = image.shape[:2]
    new_height = int(scale * height)
    return cv2.resize(image, (width, new_height))

def contrast(image):
    """Strongly reduce the contrast and shift the brightness of a cropped image, to simulate difficult lighting"""
    alpha = np.random.uniform(0.2, 0.21)
    beta = np.random.randint(-90, 90)

    # add random cropping (of the top-left corner) to the contrast function
    crop_height = np.random.randint(50, 70)
    crop_width = np.random.randint(75, 85)
    image = image[crop_height:, crop_width:]

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def elastic(image, alpha=40, sigma=6):
    """Elastic deformation: move every pixel along a smooth random displacement field (strength alpha, smoothness sigma)"""
    height, width = image.shape[:2]
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (height, width)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (height, width)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(image, x + dx, y + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def blur(image):
    """Slightly deform the image and apply a gaussian blur, to simulate blurry or imperfect captures"""
    kernel_size = np.random.choice([7, 9, 11])

    # add elastic transformation to the blurring function
    image = elastic(image, alpha=40, sigma=120 * 0.05)

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def noise(image):
    """Add strong gaussian noise, to simulate camera noise and imperfections"""
    mean = -0.5
    sigma = np.random.uniform(50, 55)
    noise = np.random.normal(mean, sigma, image.shape)
    # add the noise in float and clip it, casting negative noise to uint8 would wrap it around to bright pixels
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# add cutout function
def cutout(image):
    """Black out 11 random rectangles of about 55x55 pixels, to simulate partially hidden bottles"""
    height, width = image.shape[:2]
    mask = np.ones((height, width), np.uint8)
    num_rectangles = np.random.randint(11, 12) # choose a random number of rectangles to cut out
    for i in range(num_rectangles):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        w = np.random.randint(50, 60)
        h = np.random.randint(50, 60)
        x1 = np.clip(x - w//2, 0, width)
        y1 = np.clip(y - h//2, 0, height)
        x2 = np.clip(x + w//2, 0, width)
        y2 = np.clip(y + h//2, 0, height)
        mask[y1:y2, x1:x2] = 0
    return cv2.bitwise_and(image, image, mask=mask)


def zoom(image):
    """Zoom out: place the image at a random position on a black canvas up to 1.5 times larger, resized back to the original size"""
    height, width = image.shape[:2]

    # generate a random scale factor between 0.5 and 1.5
    scale_factor = np.random.uniform(0.5, 1.5)

    # ensure that the scale factor is greater than 1
    if scale_factor <= 1:
        scale_factor = 1.1

    # calculate the new height and width of the zoomed image
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    # check that new_width is greater than 0
    if new_width <= 0:
        new_width = 1

    # create a new image with the zoomed dimensions
    zoomed = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # calculate the offset for the zoomed image
    x_offset = np.random.randint(0, new_width - width + 1)
    y_offset = np.random.randint(0, new_height - height + 1)

    # copy the original image to the zoomed image with the offset
    zoomed[y_offset:y_offset+height, x_offset:x_offset+width] = image

    # resize the zoomed image to the original dimensions
    zoomed = cv2.resize(zoomed, (width, height))

    return zoomed


# define the augmentation functions by name
augmentations = {
    'rotate': rotate,
    'stretch': stretch,
    'luminosity': luminosity,
    'color_filter': color_filter,
    'contrast': contrast,
    'blur': blur,
    'cutout': cutout,
    'noise': noise,
    'zoom': zoom,
}

# augmentations applied by default (the set used in the report is: rotate stretch contrast blur cutout noise)
default_augmentations = ['rotate', 'stretch', 'luminosity', 'color_filter', 'blur', 'noise', 'zoom']

# create list of valid file extensions
valid_extensions = ['.jpg', '.jpeg', '.png']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmented copies of every image in a folder')
    parser.add_argument('input_dir', help='folder containing the original images')
    parser.add_argument('output_dir', help='folder to write the augmented images to')
    parser.add_argument('--augmentations', nargs='+', choices=list(augmentations), default=default_augmentations, metavar='NAME',
                        help=f"augmentations to apply to every image, among: {' '.join(augmentations)} "
                             f"(default: {' '.join(default_augmentations)})")
    parser.add_argument('--include-original', action='store_true', help='also copy the original images to the output folder')
    parser.add_argument('--seed', type=int, default=None, help='random seed, for reproducible augmentations')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # used by ColorJitter

    # create directory for augmented images
    os.makedirs(args.output_dir, exist_ok=True)

    # list the images with a valid extension
    filenames = sorted(filename for filename in os.listdir(args.input_dir)
                       if os.path.splitext(filename)[1].lower() in valid_extensions)

    # generate augmented images, named <original name>_aug_<augmentation>.jpeg
    for count, filename in enumerate(filenames):
        # load image
        img = cv2.imread(os.path.join(args.input_dir, filename))
        if img is None:
            print(f"skipping unreadable image {filename}")
            continue

        name = os.path.splitext(filename)[0]

        if args.include_original:
            shutil.copyfile(os.path.join(args.input_dir, filename), os.path.join(args.output_dir, filename))

        # generate and save augmented images
        for augmentation in args.augmentations:
            augmented_img = augmentations[augmentation](img.copy())
            cv2.imwrite(os.path.join(args.output_dir, f"{name}_aug_{augmentation}.jpeg"), augmented_img)

        if count % 100 == 0:
            print(f'{count}/{len(filenames)} images augmented...')

    print(f'done: {len(filenames)} images x {len(args.augmentations)} augmentations written to {args.output_dir}')
