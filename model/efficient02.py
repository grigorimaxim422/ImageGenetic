import torch
import torch.nn as nn
import torch.nn.functional as F

class squeeze_excitation(nn.Module):
    def __init__(self, in_c, reduction_ratio):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c//reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_c//reduction_ratio,in_c, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Sigmoid(),
        )
        

    def forward(self, x):
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.layers(x_se)
        x = x * F.sigmoid(x_se)
        return x
        # batch_size, channel_size, _, _ = x.size()
        # y = self.avgpool(x).view(batch_size, channel_size)
        # # y = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        # y = self.layers(y)
        # x = x * nn.Sigmoid(y)
        # return x
        
        # y = y.view(batch_size, channel_size, 1, 1)        
        # return x*y.expand_as(x)
    
class swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid(x)

class MBConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, t=3, reduction_ratio=1, stride=1, padding=1,  affine=True,track_running_stats=True):
        super().__init__()
        self.t = t
        self._expand_conv = nn.Sequential(
                nn.Conv2d(in_c, in_c*self.t, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_c*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.GELU()                
                )

        self._depthwise_conv = nn.Sequential(
                nn.Conv2d(in_c*self.t, in_c*self.t, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(in_c*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.GELU()
            )      
        self.se = squeeze_excitation(in_c*self.t, reduction_ratio=reduction_ratio)      
        self._project_conv = nn.Sequential(
                nn.Conv2d(in_c*self.t, out_c, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_c, affine=affine, track_running_stats=track_running_stats))
        
        self.layers = nn.Sequential(
            self._expand_conv,
            self._depthwise_conv,
            self.se,
            # swish(),
            self._project_conv
        )
            
    def forward(self, x):
        return self.layers(x)

class EfficientNet02(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNet02, self).__init__()
        self.stem_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.blocks = nn.Sequential(
            MBConv(16,16,3,2,2,1,1),
            MBConv(16,16,3,2,2,1,1),
            MBConv(16,16,3,2,2,1,1),
            MBConv(16,24,5,4,2,5,2),
            MBConv(24,24,5,4,3,1,2),
            MBConv(24,24,5,4,3,1,2),
            MBConv(24,48,5,8,6,2,2),
            MBConv(48,48,5,8,12,1,2),
            MBConv(48,108,3,13,39,2,1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(108, num_classes)

    def forward(self, x):
        x = F.gelu(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.global_pool(x)        
        x = torch.flatten(x, 1)
        x = F.dropout(x)
        x = self.fc(x)
        return x
    
    

