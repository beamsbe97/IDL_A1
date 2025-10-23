import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision
import json
import numpy as np
from utils import *
import numpy as np
import matplotlib.pyplot as plt



with open('config.json', 'r') as file:
    config = json.load(file)

transform = transforms.Compose([
    transforms.Grayscale(),              
    transforms.Resize((28, 28)),        
    transforms.ToTensor(),              
    transforms.Normalize((0.5,), (0.5,))])

trainLoader, valLoader, testLoader = get_loaders(config["dataset_name"],transform, config["batch_size"])

class MLP(nn.Module):
    def __init__(self, dropout=config["dropout"]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits

ablation="deep"
colours = ["blue", "orange", "green"]
lr = [0, 1e-3, 1e-4]
plt.figure(figsize=(10, 6))

for i in range(0, 1):
    mlp_classifier = MLP()

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(mlp_classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


    train_losses, val_losses = train_and_test(mlp_classifier, criterion, optimiser, trainLoader, valLoader, testLoader, config, ablation)
    plt.plot(train_losses, label=f"Train (Adam)", color=colours[i])
    plt.plot(val_losses, label=f"Val (Adam)", color=colours[i],linestyle="--")


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"mlp_{ablation}.png", dpi=300)
plt.close()