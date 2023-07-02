# HI my name is Ahmed TAIBECHE and this is my engineering final project: development of visual inpection system for water bottle based on deep learning (transfer learning)
# For more details see `report/PFE_report.pdf`, PS: it's in french try to use google chrome if u want to read it in english use the translator tool for the pdf, see Chapter 1, 2 and 3

# Water Bottle Visual Inspection System

## Introduction
In today's manufacturing industry, ensuring product safety is paramount, particularly in the production of bottled water. Visual inspection plays a crucial role in maintaining quality standards and safeguarding consumer health. This project focuses on the development of a robust system for visual inspection in water bottle production, leveraging deep learning techniques.

## Approach
Our approach primarily revolves around utilizing pre-trained deep learning models and employing transfer learning. Specifically, we concentrate on well-established models renowned for their performance, such as ResNet18, ResNet50, and InceptionV3. These models, having been trained on extensive datasets, offer a wealth of pre-existing knowledge which we harness through transfer learning. The objective is to retrain these networks to proficiently recognize images of both defect-free (Propre) and defective water bottles.

Through rigorous testing and evaluation, our approach achieves an impressive accuracy rate of 94.5%. These results underscore the efficacy of our system in augmenting visual inspection processes within the water bottle manufacturing industry, ultimately leading to enhanced product quality and heightened consumer satisfaction.

## Implementation Details
- **Framework**: PyTorch
- **Target Platform**: Edge application, specifically designed for deployment on Jetson Nano developer kit for inference.
- **Data Collection and Augmentation**: We collected our dataset and employed data augmentation techniques using a custom script (`data_aug.py`). The augmentations include rotation, stretching, luminosity adjustment, color filtering, blur, noise addition, and zoom.
- **Data Preprocessing**: Prior to training, we preprocessed the data, which involved resizing the images and normalization. This step was essential for ensuring consistency and optimal performance during model training.
- **Model Training**: We trained our models using PyTorch, leveraging the provided dataset. The training process was conducted via a custom script (`train.py`), optimizing the selected deep learning architectures for the task at hand.
- **Model Export**: After training, we exported the trained models to the ONNX format using (`onnx_exp.py`). This format offers compatibility and efficiency for deployment on edge devices like the Jetson Nano, facilitating seamless integration into production environments.
- **Inference**: For inference, we developed a dedicated script (`inference.py`) to deploy the trained models on the edge platform. This script enables real-time inference, allowing for rapid and accurate identification of defective water bottles during production.

## Conclusion
In conclusion, our project presents a comprehensive solution for visual inspection in water bottle production, leveraging the power of deep learning and edge computing. By integrating advanced techniques and leveraging pre-trained models, we have demonstrated significant improvements in accuracy and efficiency. We believe that our system holds great potential for revolutionizing quality control processes within the manufacturing industry, thereby enhancing product safety and consumer satisfaction.

### AHMED TAIBECHE AI , DEEP LEARNING, TRANSFER LEARNING, machine learning

### Thank you 
