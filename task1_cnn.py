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

with open('config.json', 'r') as file:
    config = json.load(file)


transform = transforms.Compose([
    transforms.Grayscale(),              
    transforms.Resize((28, 28)),        
    transforms.ToTensor(),              
    transforms.Normalize((0.5,), (0.5,))])

trainLoader, valLoader, testLoader = get_loaders(config["dataset_name"],transform, config["batch_size"])


class CNN_Classifier(nn.Module):
    def __init__(self, dropout=config["dropout"]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear_relu = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear_relu(x)
        return x

cnn_clf = CNN_Classifier()

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(cnn_clf.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

train_and_test(cnn_clf, criterion, optimiser, trainLoader, valLoader, testLoader, config, ablation)
