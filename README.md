# Project overview and setup instructions
This code implements a combined model using PyTorch for a garbage image classification task, incorporating image and text features. The approach integrates a pretrained ResNet18 model for image feature extraction and an LSTM network for processing textual descriptions, creating a robust framework to classify images based on visual and descriptive input.

Here’s a breakdown of the key components:

1. Data Preprocessing and Custom Dataset:
The GarbageDataset class loads images, labels, and description texts, converting descriptions to indices for text embedding.
Various transformations are applied to augment training data, including random cropping, rotation, color jittering, and normalization.
2. Data Loading:
Training, validation, and test data loaders are set up with batch size, shuffle, and worker configurations for efficient data handling.
3. Model Definition:
CombinedModel integrates ResNet18 for image features, an LSTM network for text features, and combines them through fully connected layers for final classification.
Dropout layers are included for regularization.
4. Training Process with Early Stopping:
The training loop records loss, accuracy, and F1 score for each epoch and incorporates early stopping to prevent overfitting.
The model saves the best performing weights based on validation loss.
5. Evaluation:
Accuracy, F1 score, confusion matrix, and classification report are computed and displayed for the test set.
6. Visualization:
Loss, accuracy, and F1 score progress over epochs are visualized.
Confusion matrix is plotted for a clear view of classification performance across classes.
How to Run:
Ensure that data paths in train_folder, val_folder, and test_folder are correctly set.
The vocabulary is shared across train, validation, and test datasets for consistent text handling.
Adjust batch size and num_workers based on hardware capabilities.
This framework is versatile for image-text combined classification tasks, with flexibility for fine-tuning various parameters and further improvements to increase model accuracy, possibly addressing any challenges in low model accuracy observed during testing.
