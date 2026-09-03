import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
import torchvision 
import torchvision.transforms as transforms 

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms)
test_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, karnel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, karnel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x 

def accuracy(model, loader):
    model.eval()
    correct = 0 
    total = 0 
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        return correct / total * 100

    