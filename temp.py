import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np


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
    
images = np.load('images.npy')      # shape: (18000, 150, 150)
labels = np.load('labels.npy')      # shape: (18000, 2)

for i in range(0,100):
    print(labels[np.random.randint(0, len(labels))])

# Convert to float32 and add channel dimension for PyTorch (C, H, W)
# images = images.astype(np.float32) / 255.0   # normalize to [0,1]
# images = np.expand_dims(images, axis=1)      # shape: (18000, 1, 150, 150)

# # Convert labels to tensors
# labels = torch.tensor(labels, dtype=torch.float32)

# dataset = ClockDataset(images, labels)

# train_size = int(0.7 * len(dataset))
# val_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - val_size

# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
# batch_size = 64

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


