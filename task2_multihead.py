import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import json
from utils import *
import numpy as np
import matplotlib.pyplot as plt

# Load config
with open('config.json', 'r') as file:
    config = json.load(file)

# Set device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define model
class MultiHeadTimeTeller(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.flatten = nn.Flatten()

        self.fc_block = nn.Sequential(
            nn.Linear(256*15*15, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.classifier_head = nn.Linear(256, 12)
        self.regressor_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_block(x)

        pred_hour = self.classifier_head(x)
        pred_min = self.regressor_head(x)
        return pred_hour, pred_min

# Initialize model and move to device
model = MultiHeadTimeTeller().to(device)

# Loss and optimizer
clf_loss = nn.CrossEntropyLoss()
reg_loss = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

# Load data
trainLoader, valLoader, testLoader = task2_get_loaders()

# Training loop
for epoch in range(config["epochs"]):
    running_train_loss = 0
    model.train()
    for index, data in enumerate(trainLoader):
        images, hours, minutes = data
        images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)

        optimiser.zero_grad()
        pred_hour, pred_min = model(images)
        loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
        loss.backward()
        optimiser.step()

        running_train_loss += loss.item()

    train_losses.append(running_train_loss / len(trainLoader))

    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(valLoader):
            images, hours, minutes = data
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)

            pred_hour, pred_min = model(images)
            loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
            running_val_loss += loss.item()

    val_losses.append(running_val_loss / len(valLoader))

# Save training data
ablation = ""    
np.save(f"training_data/task2_{ablation}_multihead_train.npy", train_losses)
np.save(f"training_data/task2_{ablation}_multihead_val.npy", val_losses)

# Evaluation
correct = 0
total = 0
model.eval()

label_hours=[]
label_minutes=[]
with torch.no_grad():
    for data in testLoader:
        images, hours, minutes = data
        images, hours, minutes = images.to(device), hours.to(device), minutes.to(device)

        outputs_hour, outputs_min = model(images)
        pred_hour = torch.max(outputs_hour, 1)
        pred_min = torch.max(outputs_min, 1)



        #concat all the batches of hours and minutes
        #concat all the batches of predicted hours and minutes
        #find mae between hours column and minutes column

        
        
        mae = torch.sqrt(nn.MSELoss())

print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
