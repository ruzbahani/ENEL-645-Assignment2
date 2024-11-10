# Project overview and setup instructions

# Garbage Image Classification with Combined Image and Text Features

This project implements a deep learning model for classifying garbage images based on both image data and textual descriptions. The model utilizes a pretrained ResNet18 for image feature extraction and an LSTM network to process text descriptions, allowing it to make more accurate classifications.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Project Overview
The primary goal of this project is to classify images of garbage based on both visual and textual data. The dataset contains images categorized into four classes: `Black`, `Green`, `Blue`, and `TTR`. Each image is labeled according to its category and includes a description that helps further define its characteristics.

### Key Features
- Uses a **pretrained ResNet18** model for image feature extraction.
- Processes text descriptions with an **LSTM network**.
- Combines features from both image and text inputs to make final predictions.
- Implements **data augmentation**, **weight initialization**, and **learning rate scheduling** to improve model performance and reduce overfitting.

## Directory Structure
```
.
├── data/
│   ├── CVPR_2024_dataset_Train/
│   ├── CVPR_2024_dataset_Val/
│   └── CVPR_2024_dataset_Test/
├── Best-Model/
│   └── best_model.pth
├── main.py               # Main code file for training and evaluation
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Requirements
The project requires the following Python libraries:
- `torch`
- `torchvision`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `Pillow`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Organize your dataset folders as follows:
   ```
   data/
   ├── CVPR_2024_dataset_Train/
   ├── CVPR_2024_dataset_Val/
   └── CVPR_2024_dataset_Test/
   ```
   Each folder should contain subfolders (`Black`, `Green`, `Blue`, `TTR`), and each subfolder should contain images of the respective category.
   
2. The model extracts image paths, labels, and descriptions based on the folder and filename structure.

## Model Architecture
The model uses a **Combined Model** class with two main components:
- **Image Branch**: ResNet18 (pretrained on ImageNet) is used to extract visual features. Layers 3 and 4 are unfrozen to allow fine-tuning on the dataset.
- **Text Branch**: A simple LSTM network processes the descriptions, embedding the text and passing it through LSTM layers to obtain textual features.

The outputs from both branches are concatenated and passed through fully connected layers for classification.

### Weight Initialization
The weights of the fully connected and LSTM layers are initialized using Xavier and Orthogonal initializations to enhance model performance and convergence.

## Training
Run the training with:
```bash
python main.py
```

### Training Parameters
- Optimizer: Adam with weight decay of `0.001`
- Learning Rate Scheduler: `StepLR` with `step_size=5` and `gamma=0.5`
- Dropout rate: `0.8` to prevent overfitting

The model saves the best-performing weights based on validation loss in the `Best-Model/` folder.

## Evaluation
The evaluation includes:
- **Accuracy** and **F1 score** for training, validation, and test sets
- **Confusion Matrix** and **Classification Report** to understand classification performance per class

After training, the model switches to evaluation mode, and predictions are compared to the actual labels on the test set.

## Results
Training and validation metrics, including loss, accuracy, and F1 scores, are visualized after each epoch to monitor model performance. The best-performing model weights are saved based on validation loss.

## Future Improvements
- **Further fine-tuning** with additional layers in ResNet.
- **Experimenting with other architectures** like MobileNet for computational efficiency.
- **Enhancing data augmentation** for better generalization.
- **Exploring attention mechanisms** to focus on relevant parts of both image and text features.
