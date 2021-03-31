# -*- coding: utf-8 -*-
import logging

from trainers.base_trainer import BaseTrainer
from utils.losses import *
from utils.metric_visualizer import LossVisualizer


class GroupUpweightingTrainer(BaseTrainer):
    """
    Simple upweighting technique which multiplies the loss by inverse group frequency.
    This has been found to work well when models are sufficiently underparameterized (e.g., low learning rate, high weight decay, fewer model parameters etc)
    Paper that investigated underparameterization with upweighting method: https://arxiv.org/abs/2005.04345
    """

    def __init__(self, option):
        super(GroupUpweightingTrainer, self).__init__(option)
        self.loss_visualizer = LossVisualizer(self.option.expt_dir)

    def get_keys_to_save(self):
        return super().get_keys_to_save() + ['group_ix_to_cnt', 'group_ix_to_weight', 'group_name_to_weight']

    def init_group_weights(self, loader):
        logging.getLogger().info("Initializing the group weights...")
        self.group_ix_to_cnt = {}
        self.group_ix_to_weight, self.group_name_to_weight = {}, {}
        self.group_ix_to_name = {}
        total_samples = 0
        for batch in loader:
            # If 'explicit_group_ix' is returned in the batch, then uses that to group the data
            # Else, uses 'group_ix' key to group the data
            if 'explicit_group_ix' in batch:
                grp_key = 'explicit_group_ix'
                grp_name_key = 'explicit_group_name'
            else:
                grp_key = 'group_ix'
                grp_name_key = 'group_name'
            for grp_ix, grp_name in zip(batch[grp_key], batch[grp_name_key]):
                grp_ix = int(grp_ix)
                if grp_ix not in self.group_ix_to_cnt:
                    self.group_ix_to_cnt[grp_ix] = 0
                self.group_ix_to_cnt[grp_ix] += 1
                self.group_ix_to_name[grp_ix] = grp_name
                total_samples += 1
        for group_ix in self.group_ix_to_cnt:
            self.group_ix_to_weight[group_ix] = total_samples / self.group_ix_to_cnt[group_ix]
            self.group_name_to_weight[self.group_ix_to_name[group_ix]] \
                = total_samples / self.group_ix_to_cnt[group_ix]

    def train(self, train_loader, test_loaders=None, unbalanced_train_loader=None):
        self.init_group_weights(train_loader)
        super().train(train_loader, test_loaders, unbalanced_train_loader)

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)

        for i, batch in enumerate(data_loader):
            batch = self.prepare_batch(batch)
            out = self.forward_model(self.model, batch)
            logits = out['logits']
            batch_losses = self.loss(out['logits'], torch.squeeze(batch['y']))
            group_key = 'explicit_group_ix' if 'explicit_group_ix' in batch else 'group_ix'

            weights = torch.FloatTensor([self.group_ix_to_weight[group_ix] for group_ix in batch[group_key]]).cuda()
            weighted_batch_losses = weights * batch_losses  # Multiply per-sample losses by weights for the corresponding groups

            self.optim.zero_grad()
            weighted_batch_losses.mean().backward()
            self.optim.step()
            self.loss_visualizer.update('Train', 'Loss', batch_losses.mean().detach().item())
            self.loss_visualizer.update('Train', 'Weighted Loss', weighted_batch_losses.mean().detach().item())
            self.update_generalization_metrics('Train', batch, weighted_batch_losses)
        self._after_train_epoch(epoch)
