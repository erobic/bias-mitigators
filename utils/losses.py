import torch
import torch.nn as nn
import torch.nn.functional as F


class GCELoss(nn.Module):
    def __init__(self, q=0.7, reduction='none'):
        super(GCELoss, self).__init__()
        self.q = q
        self.reduction = reduction

    def __call__(self, input, target):
        p = F.softmax(input, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(target, 1))
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        loss = F.cross_entropy(input, target, reduction=self.reduction) * loss_weight
        return loss


class ExpLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(ExpLoss, self).__init__()
        self.reduction = reduction

    def __call__(self, input, target):
        return torch.exp(torch.gather(1 - F.softmax(input, dim=1), dim=1, index=target.view(-1, 1)))


class InverseProbabilityLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(InverseProbabilityLoss, self).__init__()
        self.reduction = reduction

    def __call__(self, input, target):
        return 1 / torch.gather(1 - F.softmax(input, dim=1), dim=1, index=target.view(-1, 1))
        # return torch.exp(torch.gather(1 - F.softmax(input, dim=1), dim=1, index=target.view(-1, 1)))


if __name__ == "__main__":
    gce_loss = GCELoss(q=2)
    l = gce_loss(torch.FloatTensor([[0.1, 0.9], [0.1, 0.8]]), torch.LongTensor([0, 1]))
    print(l)
