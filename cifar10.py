import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# data
train_dataset = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# load pretrained resnet18
model = models.resnet18(pretrained=True)

# freeze all layers
for param in model.parameters():
    param.requires_grad = False

# replace final layer — CIFAR10 has 10 classes
model.fc = nn.Linear(512, 10)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# only train final layer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# accuracy function
def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# training loop
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accuracy = get_accuracy(model, test_loader)
    print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "cifar10_resnet.pth")
print("Model saved.")