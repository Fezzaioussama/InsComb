# Water Bottle Visual Inspection System

Visual inspection of water bottles with deep learning (transfer learning), trained with PyTorch and deployed on an NVIDIA Jetson Nano through ONNX.

**Authors:** Ahmed Taibeche & Oussama Chams Eddine Fezzai

Final-year engineering project (*Mémoire de fin d'études*, Diplôme d'Ingénieur d'État in Automatic Control and Industrial Computing), École Nationale Supérieure de Technologie, Algiers, supervised by Mme Karima Rebai and defended on 4 July 2023 (academic year 2022–2023).

The full report is in [`report/PFE_report.pdf`](report/PFE_report.pdf) (in French, see chapters 1, 2 and 3; a browser's PDF translator works well to read it in English).

## Introduction
In today's manufacturing industry, ensuring product safety is paramount, particularly in the production of bottled water. Visual inspection plays a crucial role in maintaining quality standards and safeguarding consumer health. This project focuses on the development of a robust system for visual inspection in water bottle production, leveraging deep learning techniques.

## Approach
Our approach primarily revolves around utilizing pre-trained deep learning models and employing transfer learning. Specifically, we concentrate on well-established models renowned for their performance, such as ResNet18, ResNet50, and InceptionV3. These models, having been trained on extensive datasets, offer a wealth of pre-existing knowledge which we harness through transfer learning. The objective is to retrain these networks to proficiently recognize images of both defect-free (Propre) and defective water bottles.

Through rigorous testing and evaluation, our approach achieves an accuracy of **94.5%**. These results underscore the efficacy of our system in augmenting visual inspection processes within the water bottle manufacturing industry, ultimately leading to enhanced product quality and heightened consumer satisfaction.

## Implementation Details
- **Framework**: PyTorch
- **Target Platform**: Edge application, specifically designed for deployment on the Jetson Nano developer kit for inference.
- **Data Collection and Augmentation**: We collected our dataset and employed data augmentation techniques using a custom script (`data_augmentation.py`). The augmentations include rotation, stretching, luminosity adjustment, color filtering, blur, noise addition, and zoom.
- **Data Preprocessing**: Prior to training, we preprocessed the data, which involved resizing the images and splitting them into train / validation / test sets (`data_processing_split.py`). Normalization is applied by the training transforms.
- **Model Training**: We trained our models using PyTorch on the collected dataset with a custom script (`train.py`), fine-tuning the selected pre-trained architectures for the task at hand.
- **Model Export**: After training, we exported the trained models to the ONNX format (`export_onnx.py`). This format offers compatibility and efficiency for deployment on edge devices like the Jetson Nano, facilitating seamless integration into production environments.
- **Inference**: For inference, we developed a dedicated script (`inference.py`) to deploy the trained models on the edge platform. It runs real-time inference on a camera feed, allowing rapid and accurate identification of defective water bottles during production.

## Project Structure
```
.
├── data/                       # dataset (not tracked, see below)
├── models/                     # trained checkpoints and ONNX models (not tracked)
├── report/
│   └── PFE_report.pdf          # final project report (French)
├── src/
│   ├── data/
│   │   ├── data_augmentation.py     # generate augmented images
│   │   └── data_processing_split.py # resize and split into train/val/test
│   ├── training/
│   │   ├── train.py                 # transfer-learning training loop
│   │   └── reshape.py               # replace the classifier head for N classes
│   ├── export/
│   │   ├── export_onnx.py           # PyTorch checkpoint -> ONNX
│   │   └── evaluate_onnx.py         # run an ONNX model over a folder of images
│   └── inference/
│       └── inference.py             # real-time camera inference
└── requirements.txt
```

## Getting Started

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
On the Jetson Nano, install PyTorch and `onnxruntime-gpu` from NVIDIA's Jetson wheels instead of PyPI.

### Dataset and models
The dataset (~1 GB) and the trained model (~425 MB) are too large for this repository, so `data/` and `models/` are kept empty. The training script expects one folder per class:
```
data/bottles/
├── train/
│   ├── Defective/
│   └── Propre/
└── val/
    ├── Defective/
    └── Propre/
```

### Usage
All scripts are run as modules from the repository root.

1. **Augment** the raw images of each class:
   ```bash
   python -m src.data.data_augmentation raw/Propre augmented/Propre
   python -m src.data.data_augmentation raw/Defective augmented/Defective
   ```
2. **Resize and split** each class into `train/`, `val/` and `test/`:
   ```bash
   python -m src.data.data_processing_split augmented/Propre data/bottles --class-name Propre --size 299
   python -m src.data.data_processing_split augmented/Defective data/bottles --class-name Defective --size 299
   ```
3. **Train** (use `--resolution 299` for InceptionV3, `224` for ResNet):
   ```bash
   python -m src.training.train data/bottles --arch inception_v3 --resolution 299 --model-dir models/inception_v3
   tensorboard --logdir models/inception_v3/tensorboard
   ```
4. **Export** the best checkpoint to ONNX:
   ```bash
   python -m src.export.export_onnx --model-dir models/inception_v3
   ```
5. **Evaluate** the ONNX model on the test images:
   ```bash
   python -m src.export.evaluate_onnx models/inception_v3/inception_v3.onnx data/bottles/test/Defective
   ```
6. **Run** real-time inference from a camera (press `q` to quit):
   ```bash
   python -m src.inference.inference models/inception_v3/inception_v3.onnx --camera 0
   ```

## Conclusion
Our project presents a comprehensive solution for visual inspection in water bottle production, leveraging the power of deep learning and edge computing. By integrating advanced techniques and leveraging pre-trained models, we have demonstrated significant improvements in accuracy and efficiency. We believe that our system holds great potential for improving quality control processes within the manufacturing industry, thereby enhancing product safety and consumer satisfaction.

## Acknowledgements
- Our supervisor Mme Karima Rebai and the jury members of ENST Algiers.
- The training, ONNX export and model reshape scripts are adapted from the `pytorch-classification` tools of NVIDIA's [jetson-inference](https://github.com/dusty-nv/jetson-inference).

**Keywords:** AI, deep learning, transfer learning, machine learning, computer vision, edge AI
