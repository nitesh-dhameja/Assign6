import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # First conv block - reduced channels
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 -> 8 channels
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second conv block - reduced channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8 -> 16 channels
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # First transition layer
        self.conv1x1_1 = nn.Conv2d(16, 8, kernel_size=1)  # Reduce channels to 8
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Third conv block - reduced channels
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8 -> 16 channels
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)
        
        # Second transition layer
        self.conv1x1_2 = nn.Conv2d(16, 8, kernel_size=1)  # Reduce channels to 8
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Fourth conv block - reduced channels
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8 -> 16 channels
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.1)
        
        # Third transition layer
        self.conv1x1_3 = nn.Conv2d(16, 8, kernel_size=1)  # Reduce channels to 8
        self.maxpool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3
        
        # Fifth conv block - final processing
        self.conv5 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8 -> 16 channels
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout(0.1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Reduces 3x3 -> 1x1 for each channel
        
        # Fully connected layer replaced with GAP output
        self.fc = nn.Linear(16, 10)  # Directly map 16 features to 10 classes
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.maxpool1(self.conv1x1_1(x))
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.maxpool2(self.conv1x1_2(x))
        
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        x = self.maxpool3(self.conv1x1_3(x))
        
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)
        
        x = self.gap(x)  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 16)
        x = self.fc(x)  # Map to 10 classes
        
        return x