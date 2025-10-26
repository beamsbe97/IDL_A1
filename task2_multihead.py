import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np


with open('config.json', 'r') as file:
    config = json.load(file)

# Set device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ClockDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]             # image tensor
        y_hour = self.labels[idx, 0].long()    # classification target (hour)
        y_minute = self.labels[idx, 1].float() # regression target (minute)
        return x, y_hour, y_minute

def task2_get_loaders():

    images = np.load('images.npy')      # shape: (18000, 150, 150)
    labels = np.load('labels.npy')      # shape: (18000, 2)

    # Convert to float32 and add channel dimension for PyTorch (C, H, W)
    images = images.astype(np.float32) / 255.0   # normalize to [0,1]
    images = np.expand_dims(images, axis=1)      # shape: (18000, 1, 150, 150)

    # Convert labels to tensors
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = ClockDataset(images, labels)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader

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


model = MultiHeadTimeTeller().to(device)


clf_loss = nn.CrossEntropyLoss()
reg_loss = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []


trainLoader, valLoader, testLoader = task2_get_loaders()

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
pred_hours=[]
pred_minutes=[]

with torch.no_grad():
    total_hours = 0
    correct=0
    for data in testLoader:
        images, hours, minutes = data
        images, hours, minutes = images.to(device), hours.to(device), minutes.to(device)

        outputs_hour, pred_min = model(images)
        pred_hour = torch.max(outputs_hour, 1)[1]
        total_hours = hours.size(0)
        correct+= (torch.max(outputs_hour, 1)==hours).sum().item()

        label_hours.append(hours)
        label_minutes.append(minutes)
        pred_hours.append(pred_hour)
        pred_minutes.append(pred_min)

    label_hours = torch.cat(label_hours, dim=0)
    label_minutes = torch.cat(label_minutes, dim=0)
    pred_hours = torch.cat(pred_hours, dim=0)
    pred_minutes = torch.cat(pred_minutes, dim=0)        

    mae_hours = nn.L1Loss()(pred_hours.float(), label_hours.float())
    mae_minutes = nn.L1Loss()(pred_minutes, label_minutes)
    rmse_minutes = torch.sqrt(nn.MSELoss()(pred_minutes, label_minutes))
        
    mae = mae_hours*60 + mae_minutes
    print(f"Hour accuracy: {100* correct//total_hours}")
    print(f'TotalMAE: {mae}, Hour MAE :{mae_hours}, Minutes MAE: {mae_minutes}')
    print(f"RMSE minutes: {rmse_minutes}")
