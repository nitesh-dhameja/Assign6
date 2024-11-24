import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second conv block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # Transition layer
        self.conv1x1 = nn.Conv2d(16, 8, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions and pooling
        # Input: 28x28 -> After maxpool: 14x14 -> Final conv: 14x14
        # Final channels: 16, size: 14x14
        self.fc1 = nn.Linear(16 * 14 * 14, 32)  # Changed from 7x7 to 14x14
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        batch_size = x.size(0)
        
        # First block
        x = self.conv1(x)  # 28x28 -> 28x28
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)  # 28x28 -> 28x28
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Transition layer
        x = self.conv1x1(x)  # 28x28 -> 28x28
        x = self.maxpool(x)  # 28x28 -> 14x14
        
        # Third block
        x = self.conv3(x)  # 14x14 -> 14x14
        x = F.relu(x)
        
        # Flatten and FC layers
        x = x.view(batch_size, 16 * 14 * 14)  # Updated dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 