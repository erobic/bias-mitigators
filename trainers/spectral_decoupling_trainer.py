# -*- coding: utf-8 -*-
import logging
import math
import os

import torch
from torch import nn
from torch import optim

from models.model_factory import build_model
from utils import trainer_utils
from utils.metric_visualizer import AccuracyVisualizer, LossVisualizer
from utils.metrics import Accuracy, GroupWiseAccuracy
from torch.optim import *
import json
from copy import deepcopy
from utils.trainer_utils import create_optimizer
from torch.nn import *
from utils.losses import *
from trainers.base_trainer import BaseTrainer


class SpectralDecouplingTrainer(BaseTrainer):
    """
    Implementation for:
    Pezeshki, Mohammad, et al. "Gradient Starvation: A Learning Proclivity in Neural Networks." arXiv preprint arXiv:2011.09468 (2020).

    The paper shows that decay and shift in network's logits can help decouple learning of features, which may enable learning of signal too.

    Changes from the original implementation:
    The original implementation uses this loss: = torch.log(1.0 + torch.exp(-yhat[:, 0] * (2.0 * y - 1.0)))
    However, we just use cross-entropy since the above formulation doesn't make sense when there are more than 2 classes.
    """

    def __init__(self, option):
        super(SpectralDecouplingTrainer, self).__init__(option)
        self.loss_visualizer = LossVisualizer(self.option.expt_dir)
        if self.option.spectral_decoupling_lambdas is None:
            assert self.option.spectral_decoupling_lambda is not None, 'lambda not specified'
            self.option.spectral_decoupling_lambdas = torch.ones(
                self.option.num_classes) * self.option.spectral_decoupling_lambda
        if self.option.spectral_decoupling_gammas is None:
            assert self.option.spectral_decoupling_gamma is not None, 'gammas not specified'
            self.option.spectral_decoupling_gammas = torch.ones(
                self.option.num_classes) * self.option.spectral_decoupling_gamma

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)

        for i, batch in enumerate(data_loader):
            # Forward pass
            batch = self.prepare_batch(batch)
            out = self.forward_model(self.model, batch)
            logits = out['logits']

            # Compute the prediction loss
            # The original paper uses this formulation:
            # per_sample_losses = torch.log(1.0 + torch.exp(-yhat[:, 0] * (2.0 * y - 1.0)))
            # However, we just use cross-entropy since the above formulation doesn't make sense when there are more than 2 classes.
            pred_losses = self.loss(out['logits'], torch.squeeze(batch['y']))

            per_class_lambdas = torch.FloatTensor(
                [self.option.spectral_decoupling_lambdas[y] for y in batch['y']]).cuda()
            per_class_gammas = torch.FloatTensor(
                [self.option.spectral_decoupling_gammas[y] for y in batch['y']]).cuda()

            # The loss is based on equation 28 of the paper
            # softmax = torch.softmax(logits, dim=1)
            # gt_softmax = softmax.gather(1, batch['y'].view(-1, 1)).squeeze()
            gt_logits = logits.gather(1, batch['y'].view(-1, 1)).squeeze()

            logit_l2_losses = 0.5 * per_class_lambdas * (gt_logits - per_class_gammas) ** 2
            total_losses = pred_losses + logit_l2_losses

            self.optim.zero_grad()
            total_losses.mean().backward()
            self.optim.step()

            self.loss_visualizer.update('Train', 'Total Loss', total_losses.mean().detach().item())
            self.loss_visualizer.update('Train', 'Pred Loss', pred_losses.mean().detach().item())
            self.loss_visualizer.update('Train', 'Logit L2 Loss', logit_l2_losses.mean().detach().item())

        self._after_train_epoch(epoch)
