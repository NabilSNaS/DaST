'''VGG11/13/16/19 in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Configuration dictionary for each VGG variant
# Each list defines the sequence of layers:
# - integers indicate the number of output channels in a Conv2D layer
# - 'M' indicates a MaxPooling layer
# ----------------------------
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# ----------------------------
# VGG Class Definition
# ----------------------------
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.number = 0  # internal counter (used externally if needed)

        # Create the feature extraction layers based on the chosen VGG configuration
        self.features = self._make_layers(cfg[vgg_name])

        # Optional: you can add SE modules here (commented out)
        # self.SELayer_V2 = SELayer_V2(channel=512)

        # Fully connected classifier block
        self.classifier = nn.Sequential(
            nn.Dropout(),                 # Dropout for regularization
            nn.Linear(512, 512),          # First dense layer
            nn.ReLU(True),                # Activation
            nn.Dropout(),                 # Another Dropout
            nn.Linear(512, 512),          # Second dense layer
            nn.ReLU(True),
            nn.Linear(512, 10),           # Final output layer (10 classes for CIFAR-10)
        )

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1  # Optional counter for forward passes

        # Pass input through feature extractor
        out = self.features(x)

        # Flatten the output before feeding into the classifier
        out = out.view(out.size(0), -1)

        # Pass through classifier
        out = self.classifier(out)
        return out

    def get_number(self):
        # Return the number of forward passes (optional usage)
        return self.number

    # ----------------------------
    # Helper function to create convolutional layers from config
    # ----------------------------
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # RGB input
        count = 0

        for x in cfg:
            if x == 'M':
                count += 1
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # MaxPooling
            else:
                # Conv -> BatchNorm -> ReLU
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x  # Update in_channels for the next layer

        # Append a final average pooling layer
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
