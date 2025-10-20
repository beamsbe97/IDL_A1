import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision

EPOCHS = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))]
)

batch_size = 4

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
                                            download=True, transform=transform)

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, 
                                            download=True, transform=transform)

testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

classes = {
    0:"T-shirt/top",
1 :"Trouser",
2 :"Pullover",
3 :"Dress",
4 :"Coat",
5 :"Sandal",
6 :"Shirt",
7 :"Sneaker",
8 :"Bag",
9 :"Ankle boot"
}

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits

mlp_classifier = MLP()

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(mlp_classifier.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(0, EPOCHS):
    running_loss = 0
    for index, data in enumerate(trainLoader, 0):
        inputs, labels = data

        optimiser.zero_grad()

        outputs = mlp_classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = mlp_classifier(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')