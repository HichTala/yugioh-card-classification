import numpy as np
from torch import randperm


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_episodes, n_way):
        self.n_classes = n_classes
        self.n_episodes = n_episodes
        self.n_way = n_way

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield randperm(self.n_classes)[:self.n_way]
