import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
import torchvision 
import torchvision.transforms as transforms 

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3801))])

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layer(x)
        return x 

def get_accuracy(model, loader):
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

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0 
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accuracy = get_accuracy(model, test_loader)
    print(f"Epoch {epoch + 1} / 5 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "MNIST_cnn.pth")
print("Model saved")