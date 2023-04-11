import numpy as np


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way):
        self.n_classes = n_classes
        self.n_way = n_way

    def __len__(self):
        return int(np.ceil(self.n_classes / self.n_way))

    def __iter__(self):
        for j in range(0, self.n_classes, self.n_way):
            if j + self.n_way >= self.n_classes:
                yield range(j, self.n_classes)
            yield range(j, j + self.n_way)
