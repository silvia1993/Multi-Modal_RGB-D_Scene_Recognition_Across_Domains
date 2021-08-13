import torch.nn as nn
from .networks import init_weights
class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg

        self.avgpool = nn.AvgPool2d(14, 1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, cfg.NUM_CLASSES)
        )

        init_weights(self.fc, 'normal')

    def forward(self, concat):

        x = self.avgpool(concat)
        x = x.view(x.size(0), -1)
        class_labels = self.fc(x)

        return class_labels
