import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expand_ratio=1):
        super(EfficientBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.conv2 = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.expand_ratio = expand_ratio

    def forward(self, x):
        identity = x

        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))

        if identity.shape[1] == out.shape[1]:
            out += identity
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNet, self).__init__()
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            EfficientBlock(32, 16, expand_ratio=1),
            EfficientBlock(16, 24, expand_ratio=6),
            EfficientBlock(24, 40, expand_ratio=6),
            EfficientBlock(40, 80, expand_ratio=6),
            EfficientBlock(80, 112, expand_ratio=6),
            EfficientBlock(112, 192, expand_ratio=6),
            EfficientBlock(192, 320, expand_ratio=6)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = F.relu6(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x