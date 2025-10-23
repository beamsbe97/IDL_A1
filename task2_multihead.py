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
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

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

model = MultiHeadTimeTeller()
clf_loss = nn.CrossEntropyLoss()
reg_loss = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

trainLoader, valLoader, testLoader = task2_get_loaders()

for epoch in range(0, config["epochs"]):
    running_train_loss = 0
    for index, data in enumerate(trainLoader):
        images, hours, minutes = data
        minutes = minutes.unsqueeze(1)
        optimiser.zero_grad()

        pred_hour, pred_min = model(images)
        loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
        loss.backward()
        optimiser.step()

        running_train_loss+= loss.item()

    train_losses.append(running_train_loss/len(trainLoader))

    running_val_loss = 0
    with torch.no_grad():
        for index, data in enumerate(valLoader):
            images, hours, minutes = data
            minutes = minutes.unsqueeze(1)
            pred_hour, pred_min = model(images)
            loss = clf_loss(pred_hour, hours) + reg_loss(pred_min, minutes)
            running_val_loss+= loss.item()
    val_losses.append(running_val_loss/len(valLoader))

ablation=""    
np.save(f"training_data/task2_{ablation}_multihead_train.npy", train_losses)
np.save(f"training_data/task2_{ablation}_multihead_val.npy", val_losses)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

