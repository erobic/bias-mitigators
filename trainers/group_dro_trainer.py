# -*- coding: utf-8 -*-
import logging

from trainers.base_trainer import BaseTrainer
from utils.losses import *
from utils.metric_visualizer import LossVisualizer


class GroupDROTrainer(BaseTrainer):
    """
    Implementation for:
    Sagawa, Shiori, et al. "Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization." (ICLR 2020).
    GroupDRO groups the data using explicit bias variables and class labels and optimizes for the worst-case group.
    Paper: https://arxiv.org/pdf/1911.08731.pdf
    Original codebase: https://github.com/kohpangwei/group_DRO
    """
    def __init__(self, option):
        super(GroupDROTrainer, self).__init__(option)
        self.group_weights = torch.ones(self.option.num_groups).cuda() / self.option.num_groups
        self.loss_visualizer = LossVisualizer(self.option.expt_dir)
        self.group_names = None

    def get_keys_to_save(self):
        return super().get_keys_to_save() + ['group_weights']

    def compute_group_loss_and_counts(self, losses, group_idxs):
        """
        Computes groupwise loss and count
        :param losses:
        :param group_idxs:
        :return:
        """
        group_map = (torch.LongTensor(group_idxs).cuda() == torch.arange(self.option.num_groups).unsqueeze(
            1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_group_weights_and_compute_robust_loss(self, group_loss):
        """
        Use exponent of weighted loss to update the group weights. Uses those weights to compute overall robust loss
        :param group_loss:
        :return:
        """
        self.group_weights = self.group_weights * torch.exp(self.option.group_weight_step_size * group_loss.data)
        self.group_weights = self.group_weights / (self.group_weights.sum())
        robust_loss = group_loss @ self.group_weights.detach()
        return robust_loss, self.group_weights

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)
        for i, batch in enumerate(data_loader):
            batch = self.prepare_batch(batch)
            out = self.forward_model(self.model, batch)
            logits = out['logits']
            batch_losses = self.loss(out['logits'], torch.squeeze(batch['y']))

            # Compute the GDRO loss
            group_ixs = batch[self.option.group_by]
            group_loss, _ = self.compute_group_loss_and_counts(batch_losses, group_ixs)
            robust_loss, _ = self.update_group_weights_and_compute_robust_loss(group_loss)

            self.optim.zero_grad()
            robust_loss.backward(retain_graph=True)
            self.optim.step()

            self.loss_visualizer.update('Train', 'Group Loss', group_loss.mean().detach().item())
            self.loss_visualizer.update('Train', 'Robust Loss', robust_loss.detach().item())
            self.update_generalization_metrics('Train', batch, batch_losses)

            group_name_to_weights = {}
            for gix in self.dro_group_ix_to_name:
                group_name = self.dro_group_ix_to_name[gix]
                group_name_to_weights[group_name] = float(self.group_weights[gix])

            self.loss_visualizer.update_multiple('Train Group Weights', group_name_to_weights)
            if self.option.enable_groupwise_metrics:
                self.update_groupwise_values('Train', 'Group Loss', group_loss, batch)
                self.update_groupwise_values('Train', 'Robust Loss', group_loss, batch)
        self._after_train_epoch(epoch)

    def train(self, train_loader, test_loaders=None, unbalanced_train_loader=None):
        logging.getLogger().info("Beginning the training process...")
        self.compute_max_dataset_ixs(train_loader, test_loaders)
        self._initialization()

        # Gather group names
        self.dro_group_ix_to_name = {}
        for batch in train_loader:
            for group_ix, group_name in zip(batch[self.option.group_by], batch[self.option.key_to_group_by]):
                self.dro_group_ix_to_name[group_ix] = group_name

        self._mode_setting(is_train=True)
        start_epoch = 1
        for epoch in range(start_epoch, self.option.epochs + 1):
            self._train_epoch(epoch, train_loader)
            self._after_one_epoch(epoch, test_loaders)
        self.after_all_epochs()
