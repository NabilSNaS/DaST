'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# === VGG Configuration Dictionary ===
# Defines the number of filters and max-pooling layers per VGG variant
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# === VGG Model Class ===
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.number = 0                                      # Counter to track forward passes
        self.features = self._make_layers(cfg[vgg_name])     # Build feature extractor using config

        # === Classifier (Fully Connected Layers) ===
        self.classifier = nn.Sequential(
            nn.Dropout(),                                    # Dropout for regularization
            nn.Linear(512, 512),                             # FC1
            nn.ReLU(True),
            nn.Dropout(),                                    # Dropout
            nn.Linear(512, 512),                             # FC2
            nn.ReLU(True),
            nn.Linear(512, 10),                              # FC3 to output 10 classes (e.g. CIFAR-10)
        )

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1                                 # Count number of forward passes (if used)
        out = self.features(x)                               # Extract features
        out = out.view(out.size(0), -1)                      # Flatten before FC
        out = self.classifier(out)                           # Run through classifier
        return out

    def get_number(self):
        return self.number                                   # Return number of forward calls

    # === Build Convolutional Layers from Configuration ===
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3                                      # CIFAR-10 has 3-channel RGB input
        count = 0
        for x in cfg:
            if x == 'M':
                count += 1
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # Downsampling
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # Conv layer
                    nn.BatchNorm2d(x),                                    # Batch normalization
                    nn.ReLU(inplace=True)                                 # ReLU activation
                ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]# Final average pooling
        return nn.Sequential(*layers)                         # Wrap in nn.Sequential
