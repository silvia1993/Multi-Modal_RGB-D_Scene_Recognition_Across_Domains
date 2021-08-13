import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):

    def __init__(self, cfg=None):
        super(ResNet, self).__init__()
    
        resnet_model = models.resnet18(True)

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def forward(self, x, out_keys, in_channel=3):

        out = {}
        out['0'] = self.relu(self.bn1(self.conv1(x)))
        out['1'] = self.layer1(self.maxpool(out['0']))
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])
        return [out[key] for key in out_keys]