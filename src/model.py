import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # First conv block - reduced channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)  # 1 -> 16 channels
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second conv block - reduced channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)  # 16 -> 32 channels
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.1)
        
        # First transition layer
        self.conv1x1_1 = nn.Conv2d(32, 10, kernel_size=1)  # Reduce channels to 10
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 12x12
        
        # Third conv block - reduced channels
        self.conv3 = nn.Conv2d(10, 16, kernel_size=3, padding=0)  # 10 -> 16 channels
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1) #10

        # Fourth conv block 
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=0)  # 16 -> 16 channels
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.1)  #8

        # Fifth conv block 
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=0)  # 16 -> 16 channels
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout(0.1) #6

        # Sixth conv block 
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # 16 -> 16 channels
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout(0.1) #6
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Reduces 6x6 -> 1x1 for each channel
        
        self.conv7 = nn.Conv2d(16, 10, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.maxpool1(self.conv1x1_1(x))
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.dropout5(x)

        x = self.bn6(F.relu(self.conv6(x)))
        x = self.dropout6(x)
        
        x = self.gap(x)  # Global Average Pooling
        x = self.conv7(x) 

        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 16)
        return F.log_softmax(x, dim=-1)