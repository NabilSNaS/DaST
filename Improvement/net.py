'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Small Network (Net_s): Basic CNN for MNIST-like inputs ===
class Net_s(nn.Module):
    def __init__(self):
        super(Net_s, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)        # 1 input channel (grayscale), 20 filters
        self.conv2 = nn.Conv2d(20, 50, 5, 1)       # 20 input channels -> 50 filters
        self.fc1 = nn.Linear(4*4*50, 500)          # Flattened conv output to FC
        self.fc2 = nn.Linear(500, 10)              # Final FC layer for 10-class classification

    def forward(self, x):
        x = F.relu(self.conv1(x))                  # Conv1 + ReLU
        x = F.max_pool2d(x, 2, 2)                  # MaxPool 2x2
        x = F.relu(self.conv2(x))                  # Conv2 + ReLU
        x = F.max_pool2d(x, 2, 2)                  # MaxPool 2x2
        x = x.view(-1, 4*4*50)                     # Flatten
        x = F.relu(self.fc1(x))                    # FC1 + ReLU
        x = self.fc2(x)                            # FC2 output
        return F.log_softmax(x, dim=1)             # LogSoftmax for classification

# === Medium Network (Net_m): Adds a 3rd conv layer and tracks call count ===
class Net_m(nn.Module):
    def __init__(self):
        self.number = 0                            # For tracking forward passes
        super(Net_m, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)     # Additional conv layer (3x3)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1                       # Count inference calls (used in tracking)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_number(self):
        return self.number                         # Returns how many times forward() was called

# === Large Network (Net_l): Deepest model with 4 conv layers ===
class Net_l(nn.Module):
    def __init__(self):
        super(Net_l, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)              # Reduced FC input due to pooling
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)                          # Flatten to (batch_size, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x                                    # No softmax here; use raw logits
