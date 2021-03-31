# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim

import models
import models.cnn_models
from trainers.base_trainer import BaseTrainer
from utils.trainer_utils import grad_reverse, grad_mult, create_optimizer
from torch.optim import *
import logging
from models.model_factory import build_model


class LNLTrainer(BaseTrainer):
    """
    Implementation for:
    Kim, Byungju, et al. "Learning not to learn: Training deep neural networks with biased data." (CVPR 2019).

    Method Description:
    Has a main branch taking in bias+signal to predict classes and has branches predicting bias variables.
    Bias branches act as adversaries to the main branch, thereby reducing dependencies on biases.

    Reference codebase: https://github.com/feidfoe/learning-not-to-learn

    Known Issue(s)/Confusion(s) with original repo:
    1) The paper does not provide full details to replicate the results: https://github.com/feidfoe/learning-not-to-learn/issues/7

    2) There are two forward passes, one to compute the main loss+entropy loss and the other to compute bias loss (training bias predictors).
    Unsure why it hasn't been done with single forward pass in the original repo: https://github.com/feidfoe/learning-not-to-learn/issues/5
    """

    def __init__(self, option):
        super(LNLTrainer, self).__init__(option)

    def _build_model(self):
        super()._build_model()
        self.bias_predictor = build_model(self.option,
                                          self.option.bias_predictor_name,
                                          in_dims=self.option.bias_predictor_in_dims,
                                          hid_dims=self.option.bias_predictor_hid_dims,
                                          out_dims=self.option.num_bias_classes)
        logging.getLogger().info(f"Bias predictor {self.bias_predictor}")

        if self.option.cuda:
            self.model.cuda()
            self.bias_predictor.cuda()

    def _build_optimizer(self):
        super()._build_optimizer()
        self.bias_predictor_optim = create_optimizer(self.option.optimizer_name,
                                                     named_params=self.bias_predictor.named_parameters(),
                                                     lr=self.option.lr,
                                                     weight_decay=self.option.weight_decay,
                                                     momentum=self.option.momentum,
                                                     freeze_layers=self.option.freeze_layers)

    def get_keys_to_save(self):
        return super().get_keys_to_save() + ['bias_predictor', 'bias_predictor_optim']

    def _mode_setting(self, is_train=True):
        self.model.train(is_train)
        self.bias_predictor.train(is_train)

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)

        for i, batch in enumerate(data_loader):
            # Prepare data
            batch = self.prepare_batch(batch)
            labels = batch['y']

            # Forward pass
            self.optim.zero_grad()
            self.bias_predictor_optim.zero_grad()
            out = self.forward_model(self.model, batch)
            hidden = out[self.option.bias_predictor_in_layer]
            logits = out['logits']

            bias_out = self.bias_predictor(hidden)
            bias_logits = bias_out['logits']

            # main loss
            batch_loss = self.loss(logits, torch.squeeze(labels))
            loss_pred = batch_loss.mean()
            self.loss_visualizer.update('Train', 'Loss', loss_pred.item())

            # bias loss
            bias_softmax = torch.nn.functional.softmax(bias_logits, dim=1) + 1e-8
            bias_entropy_loss = torch.mean(
                torch.sum(bias_softmax * torch.log(bias_softmax), 1)) * self.option.entropy_loss_weight
            self.loss_visualizer.update('Train', 'Bias Entropy', bias_entropy_loss.item())
            loss = loss_pred + bias_entropy_loss
            loss.backward(retain_graph=True)

            if self.option.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.option.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.bias_predictor.parameters(), self.option.grad_clip)

            self.optim.step()
            self.optim.zero_grad()
            self.bias_predictor_optim.zero_grad()

            # Train with adversarial loss
            out = self.forward_model(self.model, batch)
            hidden = out[self.option.bias_predictor_in_layer]
            hidden_feat = grad_mult(hidden, self.option.grad_reverse_factor)
            bias_out = self.bias_predictor(hidden_feat)

            bias_labels = batch[self.option.bias_variable_name]
            if isinstance(bias_labels, list):
                bias_labels = torch.LongTensor(bias_labels)
            bias_labels = bias_labels.long().cuda()
            if len(bias_labels.squeeze().shape) > 1:
                bias_labels = torch.argmax(bias_labels, dim=1).squeeze()
            bias_loss = self.loss(bias_out['logits'].squeeze(), bias_labels.squeeze())
            self.loss_visualizer.update('Train', 'Main loss', loss_pred.mean().item())
            self.loss_visualizer.update('Train', 'Bias Entropy loss', bias_entropy_loss.mean().item())
            self.loss_visualizer.update('Train', 'Bias loss', bias_loss.mean().item())
            self.update_generalization_metrics('Train', batch, batch_loss)
            bias_loss.mean().backward(retain_graph=True)

            if self.option.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.option.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.bias_predictor.parameters(), self.option.grad_clip)

            self.optim.step()
            self.bias_predictor_optim.step()

        self._after_train_epoch(epoch)

    def _after_one_epoch(self, epoch, test_loaders, force_test=False):
        super()._after_one_epoch(epoch, test_loaders, force_test)
