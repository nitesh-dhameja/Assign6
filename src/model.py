import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # First conv block - reduced initial channels
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second conv block - keep channels small
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.dropout2 = nn.Dropout(0.1)
        
        # Transition layer with more aggressive pooling
        self.conv1x1 = nn.Conv2d(8, 4, kernel_size=1)
        self.maxpool = nn.MaxPool2d(4, 4)  # More aggressive pooling
        
        # Third conv block - limited channels
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        
        # Fully connected layers with reduced dimensions
        # Input is now 7x7 due to more aggressive pooling
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
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
        x = self.maxpool(x)  # 28x28 -> 7x7 (using stride 4)
        
        # Third block
        x = self.conv3(x)  # 7x7 -> 7x7
        x = F.relu(x)
        
        # Flatten and FC layers
        x = x.view(batch_size, 8 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 