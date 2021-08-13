import torch.nn as nn
from torchvision.models.resnet import resnet18

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.cfg = cfg

        resnet = resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  

    def forward(self, source):
        
        out = {}

        out['0'] = self.relu(self.bn1(self.conv1(source)))
        out['1'] = self.layer1(out['0'])
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])
        
        self.out = out

        return out['4']