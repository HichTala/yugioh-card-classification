import torch.nn as nn


def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(input_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.reshape(x.size(0), -1)
