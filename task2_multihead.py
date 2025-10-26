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
    def __init__(self, conv_channels=[64,128,256,512,768], fc_sizes=[512,512,256], dropout=0.2):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2,2))
            in_ch = out_ch
        self.conv_block = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3,3))  # <-- ensures fixed output
        self.flatten = nn.Flatten()

        # compute in_features dynamically
        in_features = conv_channels[-1]*3*3

        fc_layers = []
        for out_features in fc_sizes:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            if dropout>0:
                fc_layers.append(nn.Dropout(dropout))
            in_features = out_features
        self.fc_block = nn.Sequential(*fc_layers)

        self.classifier_head = nn.Linear(in_features, 12)
        self.regressor_head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.adaptive_pool(x)  # <-- add adaptive pooling here
        x = self.flatten(x)
        x = self.fc_block(x)
        return self.classifier_head(x), self.regressor_head(x)




def train_and_evaluate(config, ablation_name="default"):
    model = MultiHeadTimeTeller().to(device)

    clf_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    trainLoader, valLoader, testLoader = task2_get_loaders()

    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        # --- training ---
        running_train_loss = 0
        model.train()
        for images, hours, minutes in trainLoader:
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)
            optimiser.zero_grad()
            pred_hour, pred_min = model(images)
            loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
            loss.backward()
            optimiser.step()
            running_train_loss += loss.item()
        train_losses.append(running_train_loss / len(trainLoader))

        # --- validation ---
        running_val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, hours, minutes in valLoader:
                images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)
                pred_hour, pred_min = model(images)
                loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
                running_val_loss += loss.item()
        val_losses.append(running_val_loss / len(valLoader))

    # --- save training curve ---
    np.save(f"training_data/task2_{ablation_name}_train.npy", train_losses)
    np.save(f"training_data/task2_{ablation_name}_val.npy", val_losses)

    # --- evaluation ---
    model.eval()
    label_hours, label_minutes, pred_hours, pred_minutes = [], [], [], []
    correct, total_hours = 0, 0
    with torch.no_grad():
        for images, hours, minutes in testLoader:
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)
            outputs_hour, outputs_min = model(images)
            pred_hour = torch.max(outputs_hour, 1)[1]
            correct += (pred_hour == hours).sum().item()
            total_hours += hours.size(0)
            label_hours.append(hours)
            label_minutes.append(minutes)
            pred_hours.append(pred_hour)
            pred_minutes.append(outputs_min)

    label_hours = torch.cat(label_hours)
    label_minutes = torch.cat(label_minutes)
    pred_hours = torch.cat(pred_hours)
    pred_minutes = torch.cat(pred_minutes)

    mae_hours = nn.L1Loss()(pred_hours.float(), label_hours.float())
    mae_minutes = nn.L1Loss()(pred_minutes, label_minutes)
    total_mae = mae_hours*60 + mae_minutes

    accuracy = 100 * correct / total_hours
    print(f"Ablation: {ablation_name} | Total MAE: {total_mae:.4f} | Hour acc: {accuracy:.2f}% | Hour MAE: {mae_hours:.4f} | Minutes MAE: {mae_minutes:.4f}")
    
    return total_mae.item(), accuracy



def train_and_evaluate(lr, weight_decay, conv_channels, fc_sizes, dropout, ablation_name="default"):
    model = MultiHeadTimeTeller(conv_channels=conv_channels, fc_sizes=fc_sizes, dropout=dropout).to(device)
    clf_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainLoader, valLoader, testLoader = task2_get_loaders()
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        running_train_loss = 0
        model.train()
        for images, hours, minutes in trainLoader:
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)
            optimiser.zero_grad()
            pred_hour, pred_min = model(images)
            loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
            loss.backward()
            optimiser.step()
            running_train_loss += loss.item()
        train_losses.append(running_train_loss / len(trainLoader))

        # validation
        running_val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, hours, minutes in valLoader:
                images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)
                pred_hour, pred_min = model(images)
                loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
                running_val_loss += loss.item()
        val_losses.append(running_val_loss / len(valLoader))

    # evaluation
    model.eval()
    label_hours, label_minutes, pred_hours, pred_minutes = [], [], [], []
    correct, total_hours = 0, 0
    with torch.no_grad():
        for images, hours, minutes in testLoader:
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device).unsqueeze(1)
            outputs_hour, outputs_min = model(images)
            pred_hour = torch.max(outputs_hour, 1)[1]
            correct += (pred_hour == hours).sum().item()
            total_hours += hours.size(0)
            label_hours.append(hours)
            label_minutes.append(minutes)
            pred_hours.append(pred_hour)
            pred_minutes.append(outputs_min)

    label_hours = torch.cat(label_hours)
    label_minutes = torch.cat(label_minutes)
    pred_hours = torch.cat(pred_hours)
    pred_minutes = torch.cat(pred_minutes)

    mae_hours = nn.L1Loss()(pred_hours.float(), label_hours.float())
    mae_minutes = nn.L1Loss()(pred_minutes, label_minutes)
    total_mae = mae_hours*60 + mae_minutes
    accuracy = 100 * correct / total_hours

    print(f"{ablation_name} | Total MAE: {total_mae:.4f} | Hour acc: {accuracy:.2f}% | Hour MAE: {mae_hours:.4f} | Minutes MAE: {mae_minutes:.4f}")
    
    return total_mae.item(), accuracy


lrs = [1e-2, 1e-3, 1e-4]
weight_decays = [0, 1e-4, 1e-3]
conv_options = [
    [64,128,256],
    [64,128,256,512],
    [64,128,256,512,768]
]
fc_options = [
    [512,256],
    [512,512,256]
]
dropouts = [0, 0.2, 0.5]

import itertools

results = {}

for lr, wd, conv, fc, do in itertools.product(lrs, weight_decays, conv_options, fc_options, dropouts):
    ablation_name = f"lr{lr}_wd{wd}_conv{conv}_fc{fc}_do{do}"
    total_mae, acc = train_and_evaluate(lr, wd, conv, fc, do, ablation_name)
    results[ablation_name] = {"Total MAE": total_mae, "Hour acc": acc}

# Find best model
best_model = min(results, key=lambda k: results[k]["Total MAE"])
print("Best ablation:", best_model, results[best_model])
