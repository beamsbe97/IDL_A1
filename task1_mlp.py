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
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
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

mlp_classifier = MLP()

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(mlp_classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

train_and_test(mlp_classifier, criterion, optimiser, trainLoader, valLoader, testLoader, config)
