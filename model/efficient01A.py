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

        out = F.gelu(self.bn1(self.conv1(x)))        
        out = F.gelu(self.bn2(self.conv2(out)))

        if identity.shape[1] == out.shape[1]:
            out += identity
        return out    
    
class EfficientNet01A(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNet01A, self).__init__()
        self.stem_conv = nn.Conv2d(3, 30, kernel_size=3, stride=1, padding=0)
        self.stem_bn = nn.BatchNorm2d(30)
        self.blocks = nn.Sequential(        
            
            #First Simulate
            EfficientBlock(30, 15, expand_ratio=2, stride=1,padding=1, kernel_size=3),
            EfficientBlock(15, 32, expand_ratio=7, stride=2, padding=2, kernel_size=5),
            EfficientBlock(32, 32, expand_ratio=3, stride=1, padding=2, kernel_size=5),
            EfficientBlock(32, 64, expand_ratio=7, stride=2, padding=2, kernel_size=5),
            EfficientBlock(64, 64, expand_ratio=5, stride=1, padding=2, kernel_size=5),
            EfficientBlock(64, 112, expand_ratio=10, stride=1, padding=1, kernel_size=3),
            EfficientBlock(112, 112, expand_ratio=6, stride=1, padding=1, kernel_size=3)
                                   
            
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(112, num_classes)

    def forward(self, x):
        x = F.gelu(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x