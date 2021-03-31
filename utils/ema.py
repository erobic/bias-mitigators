import torch
import numpy as np


class ClasswiseEMA:

    def __init__(self, dataset_size, alpha=0.7):
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.labels = torch.ones(dataset_size).long().cuda() * -1000
        self.parameter = torch.zeros(dataset_size).cuda()
        self.updated = torch.zeros(dataset_size).cuda()

    def update(self, data, dataset_ix, label):
        # if dataset_ix.__class__ == torch.Tensor:
        #     dataset_ix = dataset_ix.cuda()
        param = self.alpha * self.parameter[dataset_ix] + (
                1 - self.alpha * self.updated[dataset_ix]) * data
        self.updated[dataset_ix] = 1
        self.labels[dataset_ix] = label
        self.parameter[dataset_ix] = param.detach()
        return param

    def max_loss(self, label):
        label_index = torch.where(self.labels == label)[0]
        if len(label_index) == 0:
            return 1
        else:
            return self.parameter[label_index].max()


class EMA:
    def __init__(self, dataset_size, alpha=0.9):
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.parameter = torch.zeros(dataset_size).cuda()
        self.updated = torch.zeros(dataset_size).cuda()

    def update(self, data, dataset_ix):
        param = self.alpha * self.parameter[dataset_ix] + (
                1 - self.alpha * self.updated[dataset_ix]) * data
        self.updated[dataset_ix] = 1
        self.parameter[dataset_ix] = param.detach()
        return param


class WeightsEMA:
    """Exponential moving average of model parameters.
    https://anmoljoshi.com/Pytorch-Dicussions/
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
