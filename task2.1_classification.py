"""
1. Split data into 80/10/10 train/validation/test sets
2. Shuffle dataset as sample files are ordered
3. Use the smaller dataset (75x75) for initial tests
4. Produce results on larger dataset (150x150)
5. Formulate as a multi-class classification problem (for example, with 12x60=720 classes representing each minute label)
6. No matter which architecture and loss function you will use when reporting results also provide “common sense” accuracy: the absolute value of the time difference between the predicted and the actual time

Goal of this file:
Classification - treat this as a n-class classification problem. We suggest starting out with a
smaller number of categories e.g. grouping all the samples that are between [3 : 00 −3 : 30] into
a single category (results in 24 categories in total), and trying to train a CNN model. Once you
have found a working architecture, increase the number of categories by using smaller intervals
for grouping samples to increase the ’common sense’ accuracy. Can you train a network using
all 720 different labels? What problems does such a label representation have?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotter

BASE_DIR = os.getcwd() 
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the .npy files
images = np.load('./data/A1 Data/A1_Data_150/images.npy')
labels = np.load('./data/A1 Data/A1_Data_150/labels.npy')

# We use print statements to keep track of the process
print(f"Original images shape: {images.shape}")
print(f"Original labels shape: {labels.shape}")

# Normalise the images by dividing by 255 since they are images
images_normalized = images / 255.0

# Shuffle the dataset since it is ordered
indices = np.arange(images_normalized.shape[0]) # Get the indices   
np.random.shuffle(indices) # Shuffle them to get a random order of indices
images_shuffled = images_normalized[indices] # Re-order the image with the shuffled indices
labels_shuffled = labels[indices] # Re-order the labels with the shuffled indices

# Define the 80/10/10 data split using the train_test_split from scikit-learn *twice*
X_train, X_temp, y_train, y_temp = train_test_split(images_shuffled, labels_shuffled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Let's verify the split by looking at their shapes. The shapes should be 14400/1800/1800 since the total number of images is 18000
print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

def convert_labels(labels, num_classes=24):
    """
    Robust conversion of (hour, minute) -> class index.
    All logic is based on a 12-hour cycle.
    """
    hour = labels[:, 0].astype(int)
    minute = labels[:, 1].astype(int)
    hour_mod = hour % 12 # Use 12-hour cycle for all [0, 11]

    if num_classes == 24:
        # 12 hours, 2 bins/hr (30-min bins) -> [0, 23]
        # Max: (11 * 2) + 1 = 23
        class_labels = hour_mod * 2 + (minute // 30)
        
    elif num_classes == 72:
        # 12 hours, 6 bins/hr (10-min bins) -> [0, 71]
        # Max: (11 * 6) + 5 = 71
        class_labels = hour_mod * 6 + (minute // 10)
    
    elif num_classes == 120:
        # 12 hours, 10 bins/hr (6-min bins) -> [0, 119]
        # Max: (11 * 10) + 9 = 119
        class_labels = hour_mod * 10 + (minute // 6) 
    
    elif num_classes == 360:
        # 12 hours, 30 bins/hr (2-min bins) -> [0, 359]
        # Max: (11 * 30) + 29 = 359
        class_labels = hour_mod * 30 + (minute // 2)
        
    elif num_classes == 720:
        # 12 hours, 60 bins/hr (1-min bins) -> [0, 719]
        # Max: (11 * 60) + 59 = 719
        class_labels = hour_mod * 60 + minute
        
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}. Please add logic for it.")

    # This safety clip ensures no out-of-bounds errors occur
    # even if a rare rounding or data issue happens.
    class_labels = np.clip(class_labels, 0, num_classes - 1).astype(int)
    
    return class_labels

class ClockDataset(Dataset):
    
    def __init__(self, images, labels, num_classes=24):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes

        # Pre-compute all labels to classes
        self.class_labels = convert_labels(self.labels, self.num_classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and add channel dimension (C, H, W)
        image = self.images[idx]
        image_with_channel = np.expand_dims(image, axis=0) # (1, 75, 75) because PyTorch expects channel dim first
        
        # Convert image to float tensor
        image_tensor = torch.tensor(image_with_channel, dtype=torch.float32)
        
        # Get the pre-computed class label
        class_label = self.class_labels[idx]
        
        # Convert label to long tensor
        label_tensor = torch.tensor(class_label, dtype=torch.long)
        
        return image_tensor, label_tensor
    
class ClockCNN(nn.Module):
    
    def __init__(self, num_classes=24):
        super(ClockCNN, self).__init__()
        
        # Convolutional Block 1: Detect low-level features
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduce input size (75, 75) to (37, 37)
        )
        
        # Convolutional Block 2: Detect mid-level features
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduce input size (37, 37) to (18, 18)
        )

        # Convolutional Block 3: Detect high-level features
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduce input size (18, 18) to (9, 9)
        )

        # Convolutional Block 4: Deep feature extraction
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduce input size (9, 9) to (4, 4)
        )

        # Convolutional Block 5: Further deep feature extraction
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduce input size (4, 4) to (2, 2)
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduce input size (4, 4) to (2, 2)
        )

        # The flattened size is 512 (filters) * 2 * 2 (spatial dims)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 2 * 2, out_features=128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes) # Output logits
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x) 
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x) 
        x = self.classifier(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        # Move data to the specified device in case we use GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero param gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimize
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get the predictions
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def calculate_common_sense(y_true, y_pred, num_classes):
    """
    Calculates common sense error by mapping predicted class index
    back to a representative minute value.
    """
    # 1. Get true minutes from 0:00 (e.g., 11:59 -> 719)
    true_minutes = y_true[:, 0].astype(int) * 60 + y_true[:, 1].astype(int)
    
    # 2. Convert predicted class index (y_pred) back to minutes
    if num_classes == 24: # 30 min bins
        hour_bin = y_pred // 2
        min_bin = y_pred % 2
        pred_minutes = hour_bin * 60 + min_bin * 30 + 15 # +15 for center of bin
        
    elif num_classes == 72: # 10 min bins
        hour_bin = y_pred // 6
        min_bin = y_pred % 6
        pred_minutes = hour_bin * 60 + min_bin * 10 + 5 # +5 for center of bin
        
    elif num_classes == 120: # 6 min bins
        hour_bin = y_pred // 10
        min_bin = y_pred % 10
        pred_minutes = hour_bin * 60 + min_bin * 6 + 3 # +3 for center of bin
        
    elif num_classes == 360: # 2 min bins
        hour_bin = y_pred // 30
        min_bin = y_pred % 30
        pred_minutes = hour_bin * 60 + min_bin * 2 + 1 # +1 for center of bin
        
    elif num_classes == 720: # 1 min bins
        hour_bin = y_pred // 60
        min_bin = y_pred % 60
        pred_minutes = hour_bin * 60 + min_bin # No center, it *is* the minute
        
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes} in common sense calc.")

    # 3. Calculate absolute difference
    diff = np.abs(true_minutes - pred_minutes)
    
    # 4. Handle 12-hour (720 minutes) wraparound
    errors = np.minimum(diff, (12 * 60) - diff)
    
    return np.mean(errors), errors

def get_all_predictions(model, data_loader, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            
    return np.concatenate(all_preds)

def run_ablation(num_classes, epochs, lr, batch_size):
    print(f"\n--- Initializing Experiment: {num_classes} Classes ---")
    
    # Setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Datasets
    # Apply transforms only to training set
    train_dataset = ClockDataset(X_train, y_train, num_classes=num_classes)
    val_dataset = ClockDataset(X_val, y_val, num_classes=num_classes)
    test_dataset = ClockDataset(X_test, y_test, num_classes=num_classes)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = ClockCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Model created for {num_classes} classes.")

    # Training Loop 
    print("Starting model training...")
    history_data = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        history_data.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_acc})
    print("Training complete.")

    # Save Individual History 
    history_df = pd.DataFrame(history_data)
    csv_filename = os.path.join(RESULTS_DIR, f'history_cls_{num_classes}.csv')
    history_df.to_csv(csv_filename, index=False)
    print(f"\nSaved training history to '{csv_filename}'")

    # Final Evaluation 
    print("Evaluating on test and train sets for final metrics...")
    _, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Train Accuracy: {train_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Common Sense Error 
    test_predictions = get_all_predictions(model, test_loader, device)
    common_sense_error, all_errors = calculate_common_sense(
        y_test, test_predictions, num_classes=num_classes
    )
    print(f"Mean 'Common Sense' Error: {common_sense_error:.2f} minutes")

    # Save Individual Text Results
    results_filename = os.path.join(RESULTS_DIR, f'results_cls_150_{num_classes}.txt')
    with open(results_filename, 'w') as f:
        f.write(f"--- Final Test Results (Classification, {num_classes} classes) ---\n")
        f.write(f"Final Train Accuracy: {train_acc:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Mean 'Common Sense' Error: {common_sense_error:.2f} minutes\n")
    print(f"Saved final results to '{results_filename}'")

    # Generate and Save Individual Plots
    print("Generating individual plots...")
    y_test_class = convert_labels(y_test, num_classes=num_classes)
    
    plotter.plot_loss_history(
        history_df, 
        os.path.join(PLOTS_DIR, f'plot_loss_cls_150_{num_classes}.png')
    )
    plotter.plot_accuracy_history(
        history_df,
        os.path.join(PLOTS_DIR, f'plot_acc_cls_150_{num_classes}.png')
    )
    plotter.plot_confusion_matrix(
        y_test_class, test_predictions, num_classes, 
        os.path.join(PLOTS_DIR, f'plot_cm_cls_150_{num_classes}.png')
    )

    # Return summary for final plot
    return {
        'num_classes': num_classes,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'common_sense_error': common_sense_error
    }

def main():
    # Define Hyperparameters
    CLASS_LIST = [24, 72, 120, 360, 720]  # Different class counts to try
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Run ablation study  
    summary_results = []
    for num_classes in CLASS_LIST:
        results = run_ablation(
            num_classes=num_classes,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE
        )
        summary_results.append(results)
    
    # Generate Final Summary 
    print("\n--- Ablation Study Complete ---")
    summary_df = pd.DataFrame(summary_results)
    
    # Save the summary DataFrame
    summary_csv_path = os.path.join(RESULTS_DIR, 'ablation_summary_150.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved final ablation summary to '{summary_csv_path}'")
    
    # Plot the final summary
    plotter.plot_summary_results(summary_df, PLOTS_DIR)
    
    print("All tasks finished.")

if __name__ == "__main__":
    main()