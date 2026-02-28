import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class LensCNN(nn.Module):
    """
    4-block  vanilla CNN for  classification
    """

    def __init__(self, in_channels=1, num_classes=3):
        super(LensCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2)
        )

        # Block 3
        self.block3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2)
        )

        # Block 4
        self.block4 = nn.Sequential(
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Final classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Input: (B, C=1, H, W)
        Output: logits (B, num_classes)
        """

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, 1)  # (B, 256)
        x = self.classifier(x)

        return x