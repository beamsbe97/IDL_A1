import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision
import json
import numpy as np


def train_and_test(model, criterion, optimiser, trainLoader, valLoader, testLoader, config, ablation="baseline", l1_multiplier=0):
    train_losses = []
    val_losses = []
    for epoch in range(0, config["epochs"]):
        running_train_loss = 0
        for index, data in enumerate(trainLoader, 0):
            inputs, labels = data

            optimiser.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_multiplier * l1_norm

            loss.backward()
            optimiser.step()

            running_train_loss+= loss.item()

        train_losses.append(running_train_loss/len(trainLoader))

        running_val_loss = 0
        with torch.no_grad():
            for data in valLoader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss+= loss.item()

        val_losses.append(running_val_loss/len(valLoader))
    
    #save_plot(train_losses, val_losses, ablation)

    np.save(f"training_data/task1_{config['dataset_name']}_{ablation}_mlp_train_losses.npy", train_losses)
    np.save(f"training_data/task1_{config['dataset_name']}_{ablation}_mlp_val_losses.npy", val_losses)

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
    return train_losses, val_losses



def get_loaders(dataset_name, transform, batch_size):
    if dataset_name=="FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
                                                download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True, transform=transform)
    train_size = int(0.8*len(trainset))
    val_size = len(trainset)-train_size
    trainset, valset = random_split(trainset,[train_size,val_size])

    valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, 
                                                download=True, transform=transform)

    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    return trainLoader, valLoader, testLoader
def task2_get_loaders():

    images = np.load('images.npy')      # shape: (18000, 150, 150)
    labels = np.load('labels.npy')      # shape: (18000, 2)

    # Convert to float32 and add channel dimension for PyTorch (C, H, W)
    images = images.astype(np.float32) / 255.0   # normalize to [0,1]
    images = np.expand_dims(images, axis=1)      # shape: (18000, 1, 150, 150)

    # Convert labels to tensors
    labels = torch.tensor(labels, dtype=torch.float32)
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