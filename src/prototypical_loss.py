import os
import pickle

import torch
from torch.nn import functional as F


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def dist_computation(n_classes, n_supports):
    dists = []

    supports_dir = './models/pickles/supports/pickle_{}'
    queries_dir = './models/pickles/queries/pickle_{}'

    i = 0
    while True:
        supports_path = supports_dir.format(i)
        queries_path = queries_dir.format(i)

        i += 1
        if os.path.exists(supports_path) and os.path.exists(queries_path):
            with open(supports_path, "rb") as f:
                supports = pickle.load(f)
            f.close()

            dim = supports.size(-1)
            prototype = supports.view(n_classes, n_supports, dim).mean(1)
            del supports

            with open(queries_path, "rb") as f:
                queries = pickle.load(f)
            f.close()

            dists.append(euclidean_dist(queries, prototype))
            del queries
        else:
            return torch.cat(dists)


def prototypical_loss(n_classes, n_supports, n_queries):
    dists = dist_computation(n_classes, n_supports)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_queries, -1)
    label = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_queries, 1).long().cuda()

    loss_val = -log_p_y.gather(2, label).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, label.squeeze()).float().mean()

    del dists, label

    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item()
    }
