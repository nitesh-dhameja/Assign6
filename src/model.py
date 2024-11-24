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
        
        # First transition layer
        self.conv1x1_1 = nn.Conv2d(8, 4, kernel_size=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Third conv block
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.dropout3 = nn.Dropout(0.1)
        
        # Second transition layer
        self.conv1x1_2 = nn.Conv2d(8, 4, kernel_size=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Fourth conv block - final processing
        self.conv4 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        
        # Fully connected layers with reduced dimensions
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
        
        # First transition layer
        x = self.conv1x1_1(x)  # Channel reduction
        x = self.maxpool1(x)  # Spatial reduction 28x28 -> 14x14
        
        # Third block
        x = self.conv3(x)  # 14x14 -> 14x14
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Second transition layer
        x = self.conv1x1_2(x)  # Channel reduction
        x = self.maxpool2(x)  # Spatial reduction 14x14 -> 7x7
        
        # Fourth block
        x = self.conv4(x)  # 7x7 -> 7x7
        x = F.relu(x)
        
        # Flatten and FC layers
        x = x.view(batch_size, 8 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 