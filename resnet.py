'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------
# BasicBlock: Used in ResNet18/34
# --------------------------------------
class BasicBlock(nn.Module):
    expansion = 1  # Output channels multiplier (1 for BasicBlock)

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add shortcut connection
        out = F.relu(out)
        return out


# --------------------------------------
# Bottleneck: Used in ResNet50/101/152
# --------------------------------------
class Bottleneck(nn.Module):
    expansion = 4  # Output channels multiplier (4 for Bottleneck)

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 reduction
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 expansion
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # Add shortcut
        out = F.relu(out)
        return out


# --------------------------------------
# ResNet Main Class
# --------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create the 4 layers of residual blocks
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Fully connected classification head
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # Helper to create layers of residual blocks
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # First block may have stride
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # Forward pass
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # Global average pooling
        out = out.view(out.size(0), -1)  # Flatten
        out = self.linear(out)  # Fully connected output
        return out


# --------------------------------------
# Model Constructors
# --------------------------------------
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # 18 layers

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])  # 34 layers

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])  # 50 layers

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])  # 101 layers

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])  # 152 layers


# --------------------------------------
# Quick test to verify implementation
# --------------------------------------
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))  # Example input (CIFAR10 size)
    print(y.size())  # Expected output: torch.Size([1, 10])

# test()
