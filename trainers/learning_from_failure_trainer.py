# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim

import models
import models.cnn_models
from trainers.base_trainer import BaseTrainer
from utils.trainer_utils import grad_reverse
from models import model_factory
from utils.bias_retrievers import build_bias_retriever
from utils.trainer_utils import grad_mult, create_optimizer
from torch.optim import *
import logging
from models.model_factory import build_model
from utils.losses import GCELoss
import os
from utils.metrics import Accuracy, GroupWiseAccuracy
from utils.ema import ClasswiseEMA


class LffTrainer(BaseTrainer):
    """
    Implementation for:
    Nam, Junhyun, et al. "Learning from failure: Training debiased classifier from biased classifier." (NeurIPS 2020)

    The method trains a bias-only model using generalized cross entropy loss which helps models focus on easier samples, thereby amplifying the bias
    and then uses that to re-weight the samples for the main model, so that easy/biased samples receive lower weights.
    """

    def __init__(self, option):
        super(LffTrainer, self).__init__(option)

    def _build_model(self):
        super()._build_model()
        self.bias_model = build_model(self.option, self.option.model_name,
                                      out_dims=self.option.num_classes,
                                      in_dims=self.option.in_dims,
                                      hid_dims=self.option.hid_dims,
                                      freeze_layers=self.option.freeze_layers)
        logging.getLogger().info("Bias model")
        logging.getLogger().info(self.bias_model)
        self.bias_amplification_loss = GCELoss(q=self.option.bias_loss_gamma)

        if self.option.cuda:
            self.model.cuda()
            self.bias_model.cuda()
            self.loss.cuda()

    def _initialization(self):
        super()._initialization()
        self.bias_loss_ema_computer = ClasswiseEMA(self.max_dataset_ixs['Train'] + 1, alpha=self.option.bias_ema_gamma)
        self.main_loss_ema_computer = ClasswiseEMA(self.max_dataset_ixs['Train'] + 1, alpha=self.option.bias_ema_gamma)

    def _build_optimizer(self):
        super()._build_optimizer()
        self.bias_optim = create_optimizer(self.option.optimizer_name,
                                           named_params=self.bias_model.named_parameters(),
                                           lr=self.option.lr,
                                           weight_decay=self.option.weight_decay,
                                           momentum=self.option.momentum,
                                           freeze_layers=self.option.freeze_layers)

    def _mode_setting(self, is_train=True):
        self.model.train(is_train)
        self.bias_model.train(is_train)

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)
        for i, batch in enumerate(data_loader):
            batch = self.prepare_batch(batch)

            # Pass through the main model
            out = self.forward_model(self.model, batch)
            logits = out['logits']
            main_loss = self.loss(logits, batch['y'].squeeze())

            # Pass through the bias model
            bias_out = self.forward_model(self.bias_model, batch)
            bias_logits = bias_out['logits']
            bias_loss = self.loss(bias_logits, batch['y'].squeeze())

            # Update the bias and main loss EMAs (for computing weights)
            self.bias_loss_ema_computer.update(bias_loss.squeeze(),
                                               batch['dataset_ix'],
                                               batch['y'].squeeze())
            self.main_loss_ema_computer.update(main_loss.squeeze(),
                                               batch['dataset_ix'],
                                               batch['y'].squeeze())

            # Perform class-wise normalization on bias loss
            bias_loss_ema = self.bias_loss_ema_computer.parameter[batch['dataset_ix']].clone()
            main_loss_ema = self.main_loss_ema_computer.parameter[batch['dataset_ix']].clone()

            for c in range(self.option.num_classes):
                dataset_ixs_for_c = torch.where(batch['y'] == c)[0]
                if len(dataset_ixs_for_c) == 0:
                    continue
                max_bias_loss_ema = self.bias_loss_ema_computer.max_loss(c)
                bias_loss_ema[dataset_ixs_for_c] /= max_bias_loss_ema
                max_main_loss_ema = self.main_loss_ema_computer.max_loss(c)
                main_loss_ema[dataset_ixs_for_c] /= max_main_loss_ema

            # Compute sample wise weights for main model
            sample_weights = bias_loss_ema.cuda() / (bias_loss_ema.cuda() + main_loss_ema.cuda() + 1e-8)

            # Compute bias amplification loss to update the parameters of bias model
            bias_amplication_loss = self.bias_amplification_loss(bias_logits, batch['y'].squeeze())

            # Weight the main model's loss using sample weights
            main_loss = self.loss(logits, batch['y'].squeeze()) * sample_weights

            loss = main_loss.mean() + bias_amplication_loss.mean()

            self.bias_optim.zero_grad()
            self.optim.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)

            # if self.option.grad_clip is not None:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.option.grad_clip)
            #     torch.nn.utils.clip_grad_norm_(self.bias_model.parameters(), self.option.grad_clip)
            self.bias_optim.step()
            self.optim.step()

            split = 'Train'
            self.update_generalization_metrics(split + " Main", batch, main_loss)
            self.update_generalization_metrics(split + " Bias", batch, bias_loss)
            if self.option.enable_groupwise_metrics:
                self.update_groupwise_values(split, "Sample Weights", sample_weights, batch)
                self.update_groupwise_values(split, "Main Loss EMA", main_loss_ema, batch)
                self.update_groupwise_values(split, "Bias Amp Loss", bias_amplication_loss, batch)
                self.update_groupwise_values(split, "Bias Loss EMA", bias_loss_ema, batch)
        self.loss_visualizer.log(epoch, f'{split} Main')
        self.loss_visualizer.accumulate_plot_and_reset(epoch)

    def get_keys_to_save(self):
        return super().get_keys_to_save() + ['bias_model', 'bias_optim']

    def get_current_state(self):
        save_state = super().get_current_state()
        save_state['bias_loss_ema'] = self.bias_loss_ema_computer.parameter
        save_state['main_loss_ema'] = self.main_loss_ema_computer.parameter
        return save_state

    def test(self, epoch, data_key, data_loader):
        for model, model_key in [[self.model, 'Main'], [self.bias_model, 'Bias']]:
            super().test(epoch, data_key, data_loader, model=model, model_key=model_key)