import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader


# Define paths
train_dir = "./training_set/training_set"

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(15),  
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Check dataset sizes
print(f"Training samples: {len(train_dataset)}")


import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # New Conv Layer
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)  # New BatchNorm

        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Adjusted FC size
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(torch.relu(self.batch_norm4(self.conv4(x))))  # New Layer
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
model = CNN()
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / (batch_idx+1):.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {running_loss / len(train_loader):.4f}")
    
torch.save(model.state_dict(), "cat_dog_cnn.pth")
print("Model saved as cat_dog_cnn.pth")

print("Training Complete!")