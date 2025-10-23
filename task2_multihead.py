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
        self.conv_block = nn.Sequential([
            nn.Conv2d(1, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(1, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
        ])