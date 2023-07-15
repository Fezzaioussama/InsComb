"""
ONNX Runtime classifier shared by the evaluation and real-time inference scripts.
It only needs numpy, Pillow and onnxruntime (no PyTorch on the Jetson Nano), and stays
compatible with Python 3.6 (JetPack 4)
"""
import os

import numpy as np
import onnxruntime
from PIL import Image

# class names of this project, used when there is no labels.txt next to the model
DEFAULT_CLASSES = ['Defective', 'Propre']

# ImageNet statistics used to normalize the images in train.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_labels(path):
    """
    Read the class names (one per line, in model output order) written by train.py and export_onnx.py
    """
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def preprocess(image, resolution):
    """
    Same as Resize(resolution) + CenterCrop(resolution) + ToTensor() + Normalize() of the validation
    images in train.py, for an RGB PIL image. Returns a 1x3xNxN float32 array
    """
    # resize the shorter side to the resolution, keeping the aspect ratio
    width, height = image.size
    if width <= height:
        size = (resolution, int(resolution * height / width))
    else:
        size = (int(resolution * width / height), resolution)

    if image.size != size:
        image = image.resize(size, Image.BILINEAR)

    # crop the center
    left = int(round((size[0] - resolution) / 2.0))
    top = int(round((size[1] - resolution) / 2.0))
    image = image.crop((left, top, left + resolution, top + resolution))

    # scale to [0, 1], normalize, and convert from HxWxC to 1xCxHxW
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[np.newaxis])


class OnnxClassifier:
    """
    Run an exported ONNX model with the same preprocessing as the validation images in train.py
    """
    def __init__(self, model_path, labels_path=None):
        # use TensorRT / CUDA when available (Jetson Nano), fall back to the CPU otherwise
        providers = [provider for provider in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                     if provider in onnxruntime.get_available_providers()]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # the model input is 1x3xNxN, N being the resolution it was trained at
        self.resolution = self.session.get_inputs()[0].shape[-1]

        # class names from labels.txt next to the model, unless given explicitly
        if labels_path is None:
            labels_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'labels.txt')
            if not os.path.isfile(labels_path):
                labels_path = None

        self.classes = load_labels(labels_path) if labels_path else DEFAULT_CLASSES

        num_outputs = self.session.get_outputs()[0].shape[-1]
        if num_outputs != len(self.classes):
            raise ValueError(f"{model_path} has {num_outputs} outputs but {len(self.classes)} class names: {self.classes}")

    @property
    def provider(self):
        """
        Execution provider actually running the model (e.g. CUDAExecutionProvider)
        """
        return self.session.get_providers()[0]

    def __call__(self, image):
        """
        Classify an RGB PIL image, returns the predicted class name and its probability
        """
        probabilities = self.session.run(None, {self.input_name: preprocess(image, self.resolution)})[0][0]

        index = int(probabilities.argmax())
        return self.classes[index], float(probabilities[index])
