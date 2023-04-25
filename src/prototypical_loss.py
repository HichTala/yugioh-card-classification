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


def prototype_update(supports, labels, prototypes, n_way, n_supports):
    prototype_dir = './outputs/pickles/prototype/pickle_{}'

    dim = supports.size(-1)
    supports = supports.view(n_way, n_supports, dim)
    for i, label in enumerate(labels):
        prototypes[label].data = supports[i].mean(0)
    return prototypes


def prototypical_loss(outputs, labels, prototypes, n_supports, n_queries, n_way, device):
    prototypes = prototype_update(
        supports=outputs[:n_way * n_supports],
        labels=labels,
        prototypes=prototypes,
        n_way=n_way,
        n_supports=n_supports,
    )

    dist = euclidean_dist(outputs[n_way * n_supports:], prototypes)
    log_p_y = F.log_softmax(-dist, dim=1).view(n_way, n_queries, -1)

    labels = labels.view(n_way, 1, 1).expand(n_way, n_queries, 1).long().to(device)
    loss_val = -log_p_y.gather(2, labels).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item()
    }
