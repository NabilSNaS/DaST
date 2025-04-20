'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------
# Small Network (Net_s)
# ----------------------------------
class Net_s(nn.Module):
    def __init__(self):
        super(Net_s, self).__init__()
        # First convolution: input channels=1, output=20, kernel=5x5, stride=1
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # Second convolution: input=20, output=50, kernel=5x5, stride=1
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # Fully connected layer: input=4*4*50, output=500
        self.fc1 = nn.Linear(4*4*50, 500)
        # Output layer: output=10 classes
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Convolution -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # Flatten before FC layers
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # LogSoftmax for NLL loss
        

# ----------------------------------
# Medium Network (Net_m)
# ----------------------------------
class Net_m(nn.Module):
    def __init__(self):
        self.number = 0  # Internal counter (unused externally)
        super(Net_m, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)  # Extra conv layer
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1  # Track how many times forward() was called
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # LogSoftmax for classification

    def get_number(self):
        # Return the count of forward passes
        return self.number


# ----------------------------------
# Large Network (Net_l)
# ----------------------------------
class Net_l(nn.Module):
    def __init__(self):
        super(Net_l, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)  # Additional conv layer for depth
        self.fc1 = nn.Linear(50, 500)
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
        # Final feature map should be [batch_size, 50, 1, 1] -> flatten to [batch_size, 50]
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Raw logits (no softmax here)
