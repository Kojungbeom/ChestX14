import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TransitionBlock(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048):
        super(TransitionBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.avg_pool = nn.AvgPool2d(2)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.avg_pool(out)
        return out
    
class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        (N, C, H, W) = feat_map.shape

        # (N, C, 1, 1) m
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        return m + 1 / g * torch.log(area * torch.sum(
            torch.exp(g * value0), dim=(-1, -2), keepdim=True))
    

class CX_14(nn.Module):
    def __init__(self):
        super(CX_14, self).__init__()
        r50 = models.resnet50(pretrained=True)
        for param in r50.parameters():
            param.requires_grad = False
        self.layer0 = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4

        self.transition_layer = TransitionBlock()
        self.lsepool = LogSumExpPool(gamma=10)
        self.fcl = nn.Linear(2048, 8)
        
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.transition_layer(out)
        out = self.lsepool(out)
        out = out.view(out.size(0), -1)
        out = self.fcl(out)
        return out


