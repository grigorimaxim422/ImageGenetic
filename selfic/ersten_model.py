from torch.autograd import Variable
from  torch.cuda import FloatTensor
import torch.nn.functional as F
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    def __init__(self, inp=60, anim=18, kernels=1):
        super(ECABlock, self).__init__()
        self.conv1 = nn.Conv2d(inp,anim,1,1,0)
        self.conv2 = nn.Conv2d(anim,inp,1,1,0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out
        
    
class InvertedResidual(nn.Module):
    def __init__(self,inp=30, hidden=60, anim=18, outp=15, kernel_size=1, stride=1,padding=1):
        super(InvertedResidual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp,hidden,1,1,0),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden,hidden,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            ECABlock(hidden, anim),
            nn.Conv2d(hidden, outp,1,1,0),
            nn.BatchNorm2d(outp)
        )
    def forward(self, x):
        x= self.conv(x)
        return x
    
class ErstenNet(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.0):
        super(ErstenNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,30, 1, 3, padding=1),
            nn.BatchNorm2d(30),
            nn.GELU(),            
        )
                
        self.layers = nn.ModuleList()
        self.layers.append(InvertedResidual(30,60,18,15,3,1,1))
        self.layers.append(InvertedResidual(15,105,16,32,5,2,2))
        self.layers.append(InvertedResidual(32,96,18,32,5,1,2))
        self.layers.append(InvertedResidual(32,224,12,64,5,2,2))
        self.layers.append(InvertedResidual(64,320,16,64,5,1,2))
        self.layers.append(InvertedResidual(64,640,16,112,3,1,1))
        self.layers.append(InvertedResidual(112,672,8,112,3,1,1))
        self.leng=7
        # self.layers = layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(112,100);
        
    
    def forward(self, x):
        x = self.stem(x)
        # x = self.block(x)
        for c in range (self.leng):
            x = self.layers[c](x)
            
        # x = torch.mean(x, [2,3])
        x = self.dropout(x)
        x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        


# class ResBasicBlock(nn.Module):
#     def __init__(self, in_, out_, kernel_size=3, stride=1, downsample: Optional[nn.Module] = None):
#         super(ResBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)  # batchnorm
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)  # batchnorm
#         self.downsample = downsample
#         self.stride = stride
#         self.in_plane = in_
#         self.out_plane = out_

 
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)


#         if self.downsample is not None:
#           identity = self.downsample(x)
#         elif self.conv1.stride==(2,2): # if only spatial dim changes, downsample input
#           identity = nn.functional.conv2d(x,torch.ones((self.out_plane,1,1,1),device=torch.device('cuda')), bias=None, stride=2, groups=self.in_plane)


#         out += identity
#         out = self.relu(out)

#         return out