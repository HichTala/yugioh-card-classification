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


def prototypical_loss(outputs, label, n_classes, n_support, n_queries):
    dim = outputs.size(-1)

    supports = outputs[:n_classes * n_support]
    queries = outputs[n_classes * n_support:]

    prototype = supports.view(n_classes, n_support, dim).mean(1)

    dists = euclidean_dist(queries, prototype)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_queries, -1)

    loss_val = -log_p_y.gather(2, label).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, label.squeeze()).float().mean()

    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item()
    }
