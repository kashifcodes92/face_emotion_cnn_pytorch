import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        #----------- Block 1 -----------
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)

        #----------- Block 2 -----------
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)

        #----------- Block 3 -----------
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        #After 3 pools: 48, 24, 12, 6
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        #Block 1
        x = self.pool(F.relu(self.bn1(self.conv1_1(x))))

        #Block 2
        x = self.pool(F.relu(self.bn2(self.conv2_1(x))))

        #Block 3
        x = self.pool(F.relu(self.bn3(self.conv3_1(x))))    


        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x