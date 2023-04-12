import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

    def forward(self, x):
        x = self.resnet(x)
        return x.reshape(x.size(0), -1)
