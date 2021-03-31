# -*- coding: utf-8 -*-
import logging

import torch

from models import model_factory
from trainers.base_trainer import BaseTrainer
from utils.bias_retrievers import build_bias_retriever
from utils.metric_visualizer import LossVisualizer
from utils.trainer_utils import create_optimizer


class RUBiTrainer(BaseTrainer):
    """
    Adaptation of
    Cadene, Remi, et al. "Rubi: Reducing unimodal biases in visual question answering." (NeurIPS 2019).

    While the original implementation dealt with language biases in VQA by using question features as bias features,
    in this work, we use explicit labels (e.g., gender label in CelebA) as the bias input.

    Specifically, RUBi contains a main branch (taking in the full input e.g., image for CelebA or image and question for VQA)
    and a bias-only branch (e.g., taking in gender labels for CelebA).
    The sigmoid of logits from the bias-only branch are used to modulate logits from the main branch during training.

    Paper: https://arxiv.org/abs/2005.04345
    Original codebase: https://github.com/cdancette/rubi.bootstrap.pytorch
    """

    def __init__(self, option):
        super(RUBiTrainer, self).__init__(option)
        self.loss_visualizer = LossVisualizer(self.option.expt_dir)

    def _build_model(self):
        super()._build_model()
        if self.option.bias_model_name is None:
            self.option.bias_model_name = self.option.model_name
        self.bias_model = model_factory.build_model(self.option,
                                                    self.option.bias_model_name,
                                                    in_dims=self.option.bias_variable_dims,
                                                    hid_dims=self.option.bias_model_hid_dims,
                                                    out_dims=self.option.num_classes)
        logging.getLogger().info("Bias Model")
        logging.getLogger().info(self.bias_model)
        self.bias_retriever = build_bias_retriever(self.option.bias_variable_name)

        if self.option.cuda:
            self.model.cuda()
            self.bias_model.cuda()
            self.loss.cuda()

    def _build_optimizer(self):
        self.optim = create_optimizer(self.option.optimizer_name,
                                      named_params=list(self.model.named_parameters()) + list(
                                          self.bias_model.named_parameters()),
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
            self.optim.zero_grad()

            main_out = self.forward_model(self.model, batch)
            logits = main_out['logits']
            bias = self.bias_retriever(batch, main_out)

            if self.option.bias_variable_type == 'categorical':
                # If it is a categorical bias variable (e.g., gender), then convert to one hot vectors
                _bias = torch.zeros((len(bias), self.option.bias_variable_dims))
                for ix, bias_ix in enumerate(bias):
                    _bias[ix, int(bias_ix)] = 1
                bias = _bias

            bias = bias.cuda()

            # Feed into the bias model
            bias_out = self.bias_model(bias.detach())
            bias_logits = bias_out['logits']
            bias_losses = self.loss(bias_logits, batch['y'].squeeze())
            bias_loss = bias_losses.mean()

            # Modulate the main model's outputs with bias-only model's outputs
            main_losses = self.loss(logits, batch['y'].squeeze())
            loss = main_losses.mean()
            sigmoid_weight = torch.sigmoid(bias_logits)
            rubi_logits = logits * sigmoid_weight
            rubi_losses = self.loss(rubi_logits, batch['y'].squeeze())
            loss_ratio = rubi_losses / (rubi_losses + main_losses)

            # Optimize main model and bias-only model
            fused_losses = rubi_losses + bias_losses
            fused_loss = fused_losses.mean()
            fused_loss.backward(retain_graph=True)

            if self.option.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.option.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.bias_model.parameters(), self.option.grad_clip)
            self.optim.step()
            self.update_generalization_metrics('Train Main', batch, main_losses)
            self.update_generalization_metrics('Train Bias', batch, bias_losses)
            rubi_out = {}
            rubi_out['logits'] = rubi_logits
            self.update_generalization_metrics('Train RUBi', batch, rubi_losses)
            self.update_generalization_metrics('Train Fused', batch, fused_losses)
            if self.option.enable_groupwise_metrics:
                self.update_groupwise_values('RUBi Loss/(RUBi+Main)', 'Loss Ratio', loss_ratio, batch)

        self._after_train_epoch(epoch, 'Train Fused')

    def get_keys_to_save(self):
        return super().get_keys_to_save() + ['bias_model']
