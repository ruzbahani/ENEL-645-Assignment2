
# Garbage Classification Model

This project implements a model for classifying garbage images into distinct categories (Black, Green, Blue, and TTR) using a multimodal approach in PyTorch. The model combines visual data with textual descriptions, leveraging both to improve classification accuracy.

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
This project aims to develop an effective classification model for garbage sorting by using both image and text data. The model is optimized with various techniques, including data augmentation, to achieve improved accuracy and generalization.

## Directory Structure
The project code is implemented in a single script, `main.py`, which includes all necessary steps for data handling, model setup, training, and evaluation.

- `main.py`: Contains all code for loading the dataset, defining the model, training, and evaluation.
- `README.md`: Provides an overview and instructions for the project.

## Requirements
To install the required packages, clone the repository and run:
```bash
pip install -r requirements.txt
```

## Data Preparation
1. **Dataset Structure**: Images are organized into folders by category (e.g., `Black`, `Green`, `Blue`, and `TTR`).
2. **Text Descriptions**: Textual information is extracted from filenames.
3. **Data Augmentation**: The training images undergo transformations such as cropping, rotation, color jittering, and normalization to enhance generalization.

## Model Architecture
- **Image Encoder**: A ResNet18 pretrained on ImageNet fine-tuned for feature extraction from garbage images.
- **Text Encoder**: A bidirectional LSTM with an attention mechanism to emphasize relevant words in text descriptions.
- **Combined Model**: Merges image and text features through a fully connected layer to make final predictions.

## Training
- **Device Selection**: Automatically uses GPU if available; otherwise, defaults to CPU.
- **Training Process**: The model is trained using multiple epochs and techniques like early stopping to avoid overfitting.
- **Hyperparameters**: Configurations such as batch size, learning rate, and dropout rate are adjusted for optimal performance.

To train the model, run:
```bash
python main.py
```

## Evaluation
After training, the model’s performance is evaluated using accuracy and F1 score metrics across training, validation, and test sets.

- **Confusion Matrix**: Visualizes model performance across categories.
- **Classification Report**: Provides detailed precision, recall, and F1 score per class.

## Results
1. **Overall Accuracy and F1 Score**: The model's best accuracy and F1 score across different categories.
2. **Visualization**: Includes plots for training and validation losses, accuracy, and F1 scores across epochs.

## Future Improvements
- **Dataset Balancing**: Implementing advanced balancing techniques for the classes in the dataset.
- **Additional Features**: Experiment with adding more features from textual data.
- **Hyperparameter Tuning**: Further exploration of hyperparameters to enhance performance.

---

By following these steps, you can replicate the training and evaluation process for the garbage classification model and explore potential improvements for future development.
