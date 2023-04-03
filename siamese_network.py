import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, weights):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet101(weights=weights)

    def forward_once(self, x):
        output = self.resnet(x)
        return output

    def forward(self, input0, input1):
        output0 = self.forward_once(input0)
        output1 = self.forward_once(input1)

        return output0, output1
