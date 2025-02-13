
import torch
import torch.nn as nn
class ECABlock(nn.Module):
    def __init__(self, inp=60, anim=18, kernels=1):
        self.conv1 = nn.Conv2d(inp,anim,1,1,0)
        self.conv2 = nn.Conv2d(anim,inp,1,1,0)
        self.sigmoid = nn.Sigmoid()
    def forwad(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out
        
class InvertedResidual(nn.Module):
    def __init__(self,inp=30, hidden=60, anim=18, outp=15, kernel_size=3, stride=1,padding=1):
        super(InvertedResidual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp,hidden,1,1,padding=0),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden,hidden,kernels,1,1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            ECABlock(hidden, anim),
            nn.Conv2d(hidden, outp,1,1,0),
            nn.BatchNorm2d(outp)
        )
    def forward(self, x):
        x= self.conv(x)
        return x
        
        
class ErstenV2(nn.Module):
    def __init__(self, class_num=100):
        super(ErstenV2, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,30, 1, 3, padding=1),
            nn.BatchNorm2d(30),
            nn.GELU(),            
        )
        self.layers = nn.ModuleList()
        self.depth = 7
        self.layers.append(InvertedResidual(30,60,18,15,3,1))
        self.layers.append(InvertedResidual(15,105,16,32,1))
        self.layers.append(InvertedResidual(32,96,18,32,1))
        self.layers.append(InvertedResidual(32,224,12,64,1))
        self.layers.append(InvertedResidual(32,224,12,64,1))
        
        