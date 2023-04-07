import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        x = self.resnet(x)
        return x.reshape(x.size(0), -1)
