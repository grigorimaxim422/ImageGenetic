import torch
import torch.nn as nn
import torch.nn.functional as F

class squeeze_excitation(nn.Module):
    def __init__(self, in_c, reduction_ratio):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c//4, in_c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.layers(y).view(batch_size, channel_size, 1, 1)
        
        return x*y.expand_as(x)

class swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid(x)

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

        # out = F.relu6(self.bn1(self.conv1(x)))                
        # out = F.relu6(self.bn2(self.conv2(out)))
        out = F.gelu(self.bn1(self.conv1(x)))        
        out = F.gelu(self.bn2(self.conv2(out)))

        if identity.shape[1] == out.shape[1]:
            out += identity
        return out
    
class MBConv1(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_c),
            nn.BatchNorm2d(in_c),
            swish(),
            squeeze_excitation(in_c, reduction_ratio=4),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.layers(x)

class MBConv6(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, 6*in_c, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(6*in_c),
            swish(),
            nn.Conv2d(6*in_c, 6*in_c, kernel_size=kernel_size, stride=stride, groups=6*in_c, padding=padding),
            nn.BatchNorm2d(6*in_c),
            swish(),
            squeeze_excitation(6*in_c, reduction_ratio=4),
            nn.Conv2d(6*in_c, out_c, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.layers(x)
    
class EfficientNet01(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNet01, self).__init__()
        self.stem_conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.blocks = nn.Sequential(        
             #A02            
            # EfficientBlock(16, 32, expand_ratio=4, stride=2, padding=2),
            # EfficientBlock(32, 32, expand_ratio=2, stride=1, padding=2),
            # EfficientBlock(32, 64, expand_ratio=4, stride=2, padding=2),
            # EfficientBlock(64, 64, expand_ratio=3, stride=1, padding=2),
            # EfficientBlock(64, 112, expand_ratio=5, stride=1, padding=1),
            # EfficientBlock(112, 112, expand_ratio=1, stride=1, padding=1)                            
            
            #First Simulate
            # EfficientBlock(32, 16, expand_ratio=2, stride=1,padding=1, kernel_size=3),
            # EfficientBlock(16, 32, expand_ratio=7, stride=2, padding=2, kernel_size=5),
            # EfficientBlock(32, 32, expand_ratio=3, stride=1, padding=2, kernel_size=5),
            # EfficientBlock(32, 64, expand_ratio=7, stride=2, padding=2, kernel_size=5),
            # EfficientBlock(64, 64, expand_ratio=5, stride=1, padding=2, kernel_size=5),
            # EfficientBlock(64, 112, expand_ratio=10, stride=1, padding=1, kernel_size=3),
            # EfficientBlock(112, 112, expand_ratio=6, stride=1, padding=1, kernel_size=3)
            
            #A01
            EfficientBlock(32, 16, expand_ratio=2, stride=1,padding=1),
            EfficientBlock(16, 32, expand_ratio=5, stride=2, padding=1),
            EfficientBlock(32, 32, expand_ratio=2, stride=1, padding=1),
            EfficientBlock(32, 64, expand_ratio=5, stride=2, padding=1),
            EfficientBlock(64, 64, expand_ratio=3, stride=1, padding=1),
            EfficientBlock(64, 112, expand_ratio=2, stride=1, padding=1),
            EfficientBlock(112, 112, expand_ratio=3, stride=1, padding=1)
           
            
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