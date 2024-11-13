import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn.init as init
from tqdm import tqdm

# Setting up the device for computation (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'I will use {device}')

# Define the Custom Dataset for garbage image classification
# The dataset includes images, labels, and descriptions
class GarbageDataset(Dataset):
    def __init__(self, folder_path, transform=None, vocab=None, max_description_len=5):
        self.folder_path = folder_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.descriptions = []
        self.max_description_len = max_description_len
        self.vocab = vocab or {}
        
        label_dict = {'Black': 0, 'Green': 1, 'Blue': 2, 'TTR': 3}
        
        for label_name, label_idx in label_dict.items():
            label_folder = os.path.join(folder_path, label_name)
            if not os.path.isdir(label_folder):
                continue
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                self.data.append(img_path)
                self.labels.append(label_idx)
                
                # Extracting descriptions from the filename and padding to a fixed length
                description = filename.split('.')[0].split('_')
                description_indices = [self._word_to_idx(word) for word in description]
                
                if len(description_indices) < self.max_description_len:
                    description_indices += [0] * (self.max_description_len - len(description_indices))
                else:
                    description_indices = description_indices[:self.max_description_len]
                    
                self.descriptions.append(description_indices)

    def _word_to_idx(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        description_indices = self.descriptions[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(description_indices), torch.tensor(label)



# Data path in the system for use in the program
#I have set the data paths to local directories on my hard drive. Please change the data paths to your desired directories before testing the code.
#The default dataset path : /work/TALC/enel645_2024f/garbage_data
base_path = 'C:/ali/garbage_data'  
train_folder = os.path.join(base_path, 'CVPR_2024_dataset_Train')
val_folder = os.path.join(base_path, 'CVPR_2024_dataset_Val')
test_folder = os.path.join(base_path, 'CVPR_2024_dataset_Test')


# Define a series of transformations to apply to the training images
torchvision_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torchvision_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoaders
train_dataset = GarbageDataset(train_folder, transform=torchvision_transform)
val_dataset = GarbageDataset(val_folder, transform=torchvision_transform_test, vocab=train_dataset.vocab)
test_dataset = GarbageDataset(test_folder, transform=torchvision_transform_test, vocab=train_dataset.vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define batch size and number of workers for data loading (adjust as needed for system performance)
batch_size = 64
num_workers = 4

# Get a batch of images
data_iter = iter(train_loader)
images, descriptions, labels = next(data_iter)

# Display a few images with applied augmentations
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img = images[i].numpy().transpose((1, 2, 0))
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Revert normalization
    img = np.clip(img, 0, 1)
    axs[i].imshow(img)
    axs[i].axis('off')
plt.show()

# Define class names for each types of garbage (categories based on folder names)
class_names = ['Black', 'Green', 'Blue', 'TTR']

# Create a dictionary that maps each class name to a unique index (e.g., 'Black' -> 0, 'Green' -> 1, etc.)
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Create a reverse dictionary that maps each index back to the class name (e.g., 0 -> 'Black', 1 -> 'Green', etc.)
idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}

# Print the list of class names to verify the setup
print(class_names)

# Calculate and print the total number of images in the training set
print("Train set:", len(train_loader) * batch_size)

# Calculate and print the total number of images in the validation set
print("Val set:", len(val_loader) * batch_size)

# Calculate and print the total number of images in the test set
print("Test set:", len(test_loader) * batch_size)

# Extract images, labels, and text descriptions from a given folder
def extract_data_from_folders(base_dir):
    data = []

    # Traverse through each subfolder
    for label_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, label_folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Loop through each image file in the subfolder
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filter image files
                    image_path = os.path.join(folder_path, filename)

                    # Extract text from filename (remove file extension)
                    text_description = os.path.splitext(filename)[0]

                    # Append image path, text, and label to the data list
                    data.append({
                        'image_path': image_path,
                        'text_description': text_description,
                        'label': label_folder  # The subfolder name represents the label (bin)
                    })

    # Convert to DataFrame for easy manipulation
    return pd.DataFrame(data)

# Extract the data
trainset_df = extract_data_from_folders(train_folder)
valset_df = extract_data_from_folders(val_folder)
testset_df = extract_data_from_folders(test_folder)


# Train Set
print (trainset_df['label'].value_counts())

# Validation Set
print (valset_df['label'].value_counts())

# Test Set
print (testset_df['label'].value_counts())

# Aggregate the count of labels from each dataset
train_counts = trainset_df['label'].value_counts().sort_index()
val_counts = valset_df['label'].value_counts().sort_index()
test_counts = testset_df['label'].value_counts().sort_index()

# Create a bar chart
fig, ax = plt.subplots()
index = range(len(train_counts))
bar_width = 0.25

ax.bar(index, train_counts, bar_width, label='Train')
ax.bar([p + bar_width for p in index], val_counts, bar_width, label='Validation')
ax.bar([p + bar_width * 2 for p in index], test_counts, bar_width, label='Test')

ax.set_xlabel('Class')
ax.set_ylabel('Number of samples')
ax.set_title('Class distribution across datasets')
ax.set_xticks([p + bar_width for p in index])
ax.set_xticklabels(train_counts.index)
ax.legend()

plt.show()

train_iterator = iter(train_loader)
train_batch = next(train_iterator)
print(train_batch[0].size())
print(train_batch[1].size())

plt.figure()
plt.imshow(train_batch[0].numpy()[10].transpose(1,2,0))
plt.show()

# Define a function to initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Using Xavier initialization for Linear layers
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        # Using Xavier initialization for LSTM weights
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.constant_(param.data, 0)

class CombinedModel(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim=128, hidden_dim=266, num_classes=4, dropout_rate=0.6):
        super(CombinedModel, self).__init__()
        
        # Initialize the pretrained ResNet18 model for image processing
        self.image_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the fully connected layer to match the hidden dimension
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, hidden_dim)
        # Freeze all parameters of the model to prevent them from being updated during training
        for param in self.image_model.parameters():
            param.requires_grad = False
        # Unfreeze the parameters of layer4 to allow training on them
        for param in self.image_model.layer4.parameters():
            param.requires_grad = True

        # Setup for text processing: embedding layer, BiLSTM, and Attention mechanism
        self.embedding = nn.Embedding(vocab_sizes, embedding_dim)  # Embedding layer for textual input
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)  # Bi-directional LSTM for capturing context
        self.text_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Linear layer to process LSTM output
        self.attention = nn.Linear(hidden_dim * 2, 1)  # Attention layer to compute importance weights
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer to prevent overfitting

        # Fully connected layer to combine features from image and text and classify them
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, image, description):
        # Extract features from the image using the pretrained ResNet18
        img_features = self.image_model(image)
        
        # Process the text description
        description = self.embedding(description)  # Convert text indices to embeddings
        lstm_out, _ = self.lstm(description)  # Pass embeddings through LSTM
        
        # Apply attention mechanism to the LSTM outputs
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        text_features = (lstm_out * attention_weights).sum(dim=1)  # Weighted sum of LSTM outputs
        
        # Process text features through a linear layer and apply dropout
        text_features = self.text_fc(text_features)
        text_features = self.dropout(text_features)

        # Concatenate image and text features and pass through a dropout layer
        combined_features = torch.cat((img_features, text_features), dim=1)
        combined_features = self.dropout(combined_features)
        
        # Output the final classification result
        output = self.fc(combined_features)
        return output


# Visualization functions for model training outcomes
def plot_confusion_matrix(conf_matrixs, class_namess):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrixs, annot=True, fmt="d", cmap="Blues", xticklabels=class_namess, yticklabels=class_namess)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_validation_loss(training_lossess, validation_lossess):
    plt.figure(figsize=(10, 6))
    plt.plot(training_lossess, label="Training Loss")
    plt.plot(validation_lossess, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()   

def plot_training_validation_accuracy(training_accuraciess, validation_accuraciess):
    plt.figure(figsize=(10, 6))
    plt.plot(training_accuraciess, label="Training Accuracy")
    plt.plot(validation_accuraciess, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.show()    

def plot_training_validation_f1_score(training_f1_scoress, validation_f1_scoress):
    plt.figure(figsize=(10, 6))
    plt.plot(training_f1_scoress, label="Training F1 Score")
    plt.plot(validation_f1_scoress, label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score Over Epochs")
    plt.legend()
    plt.show()
    

# Reinitialize model with new dropout rate
vocab_size = len(train_dataset.vocab)
model = CombinedModel(vocab_size).to(device)

# Update optimizer with increased weight decay
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0006)

# Define loss function (criterion) for classification
criterion = nn.CrossEntropyLoss()

# Add a learning rate scheduler to adjust the learning rate during training
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.65)

# Training loop with early stopping and performance tracking
# Set the number of epochs for training and early stopping parameters
num_epochs = 15
patience = 2  # Define patience for early stopping: number of epochs to wait if no improvement
best_val_loss = float('inf')  # Initialize best validation loss for early stopping comparison
trigger_times = 0  # Counter for how many times the validation loss has not improved


# Lists to store performance metrics for plotting later
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []
training_f1_scores = []
validation_f1_scores = []

# Training loop (Main)
for epoch in range(num_epochs):
        
    # Show Progress Bar (current epoch number)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    
    # Set model to training mode
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    all_train_labels = []
    all_train_predictions = []
        
    # Iterate over each batch in the training loader
    for images, descriptions, labels in tqdm(train_loader, desc="Training", leave=True):
        # Move tensors to the appropriate device (GPU or CPU)
        images, descriptions, labels = images.to(device), descriptions.to(device), labels.to(device)
        
        # Forward pass: compute predicted outputs by passing inputs to the model
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        outputs = model(images, descriptions)
        loss = criterion(outputs, labels) # Calculate the batch loss
        
        # Backward and optimize
        # compute gradient of the loss with respect to model parameters
        loss.backward()
        # Perform a single optimization step (parameter update)
        optimizer.step()
        
        # Accumulate loss and calculate training accuracy for the epoch
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # Collect labels and predictions for F1 Score
        all_train_labels.extend(labels.cpu().numpy())
        all_train_predictions.extend(predicted.cpu().numpy())
    
    # Calculate training metrics:Compute training metrics for the current epoch
    training_loss = running_loss / len(train_loader)
    training_losses.append(training_loss)
    
    training_accuracy = correct_train / total_train
    training_accuracies.append(training_accuracy)
    
    training_f1_score = f1_score(all_train_labels, all_train_predictions, average='weighted')
    training_f1_scores.append(training_f1_score)
    
    # Validation Phase:Switch model to evaluation mode for validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_val_labels = []
    all_val_predictions = []
    
    # No gradient updates during validation, to evaluate model performance    
    with torch.no_grad():
        for images, descriptions, labels in tqdm(val_loader, desc="Validation", leave=True):
            images, descriptions, labels = images.to(device), descriptions.to(device), labels.to(device)   
            outputs = model(images, descriptions)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy for validation data
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
            # Calculate F1 Score evaluation: Collect labels and predictions for F1 Score
            all_val_labels.extend(labels.cpu().numpy())
            all_val_predictions.extend(predicted.cpu().numpy())
    
    # Calculate validation metrics
    validation_loss = val_loss / len(val_loader)
    validation_losses.append(validation_loss)
    
    validation_accuracy = correct_val / total_val
    validation_accuracies.append(validation_accuracy)
    
    validation_f1_score = f1_score(all_val_labels, all_val_predictions, average='weighted')
    validation_f1_scores.append(validation_f1_score)
    
    # Print metrics for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, "
          f"Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}, "
          f"Training F1 Score: {training_f1_score:.4f}, Validation F1 Score: {validation_f1_score:.4f}")
    
    # Update learning rate
    scheduler.step()
    
    # Early Stopping check
    # if the validation loss has not improved-->Early Stopping
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss # Update the best validation loss
        trigger_times = 0  # Reset trigger times
        # Save the best model
        torch.save(model.state_dict(), './Best-Model/best_model.pth')  
    else:
        # Increment the trigger count
        trigger_times += 1 
        print(f"Early stopping trigger times: {trigger_times}")

        # If the number of triggers equals the patience, stop the training        
        if trigger_times >= patience:
            print("Early stopping")
            break
            
# Final message to indicate training is complete
print("Training complete.")
print("######################################################")
print("Reports:")

# Plot performance metrics using the defined functions
plot_training_validation_loss(training_losses, validation_losses)
plot_training_validation_accuracy(training_accuracies, validation_accuracies)
plot_training_validation_f1_score(training_f1_scores, validation_f1_scores)


# Switch to evaluation mode
model.eval()

# Lists to store all true labels and predictions
all_labels = []
all_predictions = []

# No gradient computation for testing
# Disable gradient computation to reduce memory consumption and speed up computations
with torch.no_grad():
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total predictions

    # Iterate through the test dataset
    for images, descriptions, labels in test_loader:

        # Move tensors to the appropriate device (GPU or CPU)
        images, descriptions, labels = images.to(device), descriptions.to(device), labels.to(device)
        
        # Forward pass (Perform a forward pass to compute predictions)
        outputs = model(images, descriptions)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate total and correct predictions for accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Collect all labels and predictions
        # Append labels and predictions to the lists for detailed analysis        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate the accuracy of the model on the test dataset
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate the weighted F1 Score to take class imbalance into account    
    # using 'weighted' average to account for class imbalance
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1 Score (Weighted): {f1:.4f}")

    
    # Print a detailed classification report
    class_report = classification_report(all_labels, all_predictions)
    print("Classification Report:")
    print(class_report)
        
    # Compute and display the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    #Visualize the confusion matrix for a clear presentation of model performance
    print("Confusion Matrix:")
    class_names = ['Black', 'Green', 'Blue', 'TTR']
    plot_confusion_matrix(conf_matrix, class_names)
    
    

