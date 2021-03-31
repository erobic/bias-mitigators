# -*- coding: utf-8 -*-
import logging
import math
import os

from models.model_factory import build_model
from utils import trainer_utils
from utils.metric_visualizer import AccuracyVisualizer, LossVisualizer
from utils.metrics import Accuracy, GroupWiseAccuracy
from torch.optim import *
import json
from utils.trainer_utils import create_optimizer
from torch.nn import *
from utils.losses import *
import numpy as np
from torch.utils.data import Dataset, Subset
from eval.vqa.gqa_eval import GQAEval


class BaseTrainer(object):
    def __init__(self, option):
        self.option = option
        self._build_model()
        self._build_optimizer()
        self.accuracy_visualizer = AccuracyVisualizer(self.option.expt_dir)
        self.loss_visualizer = LossVisualizer(self.option.expt_dir)
        self.max_dataset_ixs = {}
        self.metrics = {}  # dictionary: epoch -> data split (e.g., val, test) -> metrics
        self.group_ix_to_name = {}

    def _build_model(self):
        """
        Constructs the model using the model factory
        :return:
        """
        self.model = build_model(
            self.option,
            self.option.model_name,
            out_dims=self.option.num_classes,
            in_dims=self.option.in_dims,
            hid_dims=self.option.hid_dims,
            freeze_layers=self.option.freeze_layers)
        logging.getLogger().info(f"Model {self.model}")
        self.loss = eval(self.option.loss_type)(reduction='none')

        if self.option.cuda:
            self.model.cuda()
            self.loss.cuda()

    def _build_optimizer(self, lr=None, weight_decay=None, named_params=None):
        if lr is None:
            lr = self.option.lr
        if weight_decay is None:
            weight_decay = self.option.weight_decay
        if named_params is None:
            named_params = self.model.named_parameters()
        self.optim = create_optimizer(self.option.optimizer_name,
                                      named_params=named_params,
                                      lr=lr,
                                      weight_decay=weight_decay,
                                      momentum=self.option.momentum,
                                      freeze_layers=self.option.freeze_layers,
                                      custom_lr_config=self.option.custom_lr_config)

    def _initialization(self):
        if self.option.load_checkpoint is not None:
            self.load(self.option.load_checkpoint)
            logging.getLogger().info(f"Loaded from {self.option.load_checkpoint}")

    def _mode_setting(self, is_train):
        self.is_train = is_train
        self.model.train(is_train)

    def prepare_batch(self, batch):
        if 'gqa' in self.option.dataset_name.lower():
            batch['frcn_feat'] = batch['frcn_feat'].cuda()
            batch['bbox_feat'] = batch['bbox_feat'].cuda()
            batch['question_token_ixs'] = batch['question_token_ixs'].cuda()
        else:
            batch['x'] = batch['x'].cuda()
        batch['dataset_ix'] = torch.LongTensor(batch['dataset_ix'])
        batch['y'] = torch.LongTensor(batch['y']).cuda()
        if 'gqa' in self.option.dataset_name.lower():
            pass
        return batch

    def compute_loss(self, out, labels):
        batch_losses = self.loss(out['logits'], torch.squeeze(labels))
        return batch_losses

    def forward_model(self, model, batch, model_in=None):
        if model_in is not None:
            return model(model_in)
        if 'gqa' in self.option.dataset_name.lower():
            return model(batch['frcn_feat'], batch['bbox_feat'], batch['question_token_ixs'])
        else:
            return model(batch['x'])

    def before_train(self, train_loader, test_loaders, compute_max_dataset_ixs=True, test_load_checkpoint=True):
        logging.getLogger().info("Beginning the training process...")
        if compute_max_dataset_ixs:
            self.compute_max_dataset_ixs(train_loader, test_loaders)
        self._initialization()
        if test_load_checkpoint and self.option.load_checkpoint is not None:
            logging.getLogger().info("Evaluating immediately after loading the checkpoint")
            self._after_one_epoch(-1, test_loaders, force_test=True)
        self._mode_setting(is_train=True)

    def train(self, train_loader, test_loaders=None, unbalanced_train_loader=None):
        self.before_train(train_loader, test_loaders)
        start_epoch = 1
        for epoch in range(start_epoch, self.option.epochs + 1):
            self._train_epoch(epoch, train_loader)
            self._after_one_epoch(epoch, test_loaders)
        self.after_all_epochs()

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)
        for i, batch in enumerate(data_loader):
            batch = self.prepare_batch(batch)
            self.optim.zero_grad()
            out = self.forward_model(self.model, batch)
            loss_pred = self.loss(out['logits'], torch.squeeze(batch['y']))
            loss_pred.mean().backward(retain_graph=True)
            self.optim.step()
            self.update_generalization_metrics('Train', batch, loss_pred)

        self.optim.zero_grad()
        self._after_train_epoch(epoch)

    def _after_train_epoch(self, epoch, split='Train'):
        self.loss_visualizer.log(epoch, split)
        self.loss_visualizer.accumulate_plot_and_reset(epoch)

    def test(self, epoch, data_key, data_loader, model=None, model_key="Main"):
        if 'gqa' in self.option.dataset_name.lower():
            self.test_gqa(epoch, data_key, data_loader, model, model_key)
        else:
            self.test_default(epoch, data_key, data_loader, model, model_key)

    def update_groupwise_values(self, split, suffix, sample_wise_vals, batch):
        group_values = {}
        # Any key in batch that contains both 'group' and 'name' substrings will be used for evaluation
        for key in batch:
            if key == 'group_name':  # 'group' in key and 'name' in key:
                for val, grp_name in zip(sample_wise_vals, batch[key]):
                    if grp_name not in group_values:
                        group_values[grp_name] = []
                    group_values[grp_name].append(val.item())

        for grp_name in group_values:
            self.loss_visualizer.update(split + " " + suffix, grp_name, group_values[grp_name])

    def get_keys_to_save(self):
        # Override this method to persist variables in the trainer instance
        return ['model', 'optim', 'metrics']

    def get_current_state(self):
        keys = self.get_keys_to_save()
        save_state = {}
        for key in keys:
            if hasattr(self, key) and getattr(self, key) is not None:
                attr = getattr(self, key)
                if hasattr(attr, 'state_dict'):
                    save_state[key] = attr.state_dict()
                else:
                    save_state[key] = attr
        return save_state

    def compute_max_dataset_ixs(self, train_loader, test_loaders):
        if 'gqa' in self.option.dataset_name.lower():
            self.max_dataset_ixs['Train'] = 943000
            self.max_dataset_ixs['Val All'] = 51044
            self.max_dataset_ixs['Val Head'] = 33881
            self.max_dataset_ixs['Val Tail'] = 17162
            self.max_dataset_ixs['Test Dev All'] = 2795
            self.max_dataset_ixs['Test Dev Head'] = 1732
            self.max_dataset_ixs['Test Dev Tail'] = 1062
            self.max_dataset_ixs['Val non-OOD'] = 132062
            self.max_dataset_ixs['Test Dev non-OOD'] = 12578
            return
        elif 'celeba' in self.option.dataset_name.lower():
            self.max_dataset_ixs['Train'] = 1627690
            self.max_dataset_ixs['Val'] = 182636
            self.max_dataset_ixs['Test'] = 202598
        else:
            for key in test_loaders:
                logging.getLogger().info(f"Computing max dataset ix for {key}")
                self.compute_max_dataset_ix(key, test_loaders[key])

    def compute_max_dataset_ix(self, key, loader):
        self.group_ix_to_name = {}
        self.group_name_to_ix = {}
        if key not in self.max_dataset_ixs:
            self.max_dataset_ixs[key] = -1
        for batch in loader:
            mdi = int(max(batch['dataset_ix']))
            if mdi > self.max_dataset_ixs[key]:
                self.max_dataset_ixs[key] = mdi

            if 'group_name' in batch:
                for group_ix, group_name in zip(batch['group_ix'], batch['group_name']):
                    self.group_ix_to_name[group_ix] = group_name
                    self.group_name_to_ix[group_name] = group_ix
        logging.getLogger().info(f"max dataset ix for {key} = {self.max_dataset_ixs[key]}")
        return self.max_dataset_ixs[key]

    def save(self, save_file):
        save_dir = trainer_utils.get_dir(save_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_state = self.get_current_state()
        torch.save(save_state, save_file)
        logging.getLogger().info(f"Saved to {save_file}")

    def load(self, ckpt_path):
        save_state = torch.load(ckpt_path)
        keys = self.get_keys_to_save()
        for key in keys:
            if hasattr(self, key) and getattr(self, key) is not None and key in save_state:
                prop = getattr(self, key)
                if hasattr(prop, 'load_state_dict'):
                    prop.load_state_dict(save_state[key])
                else:
                    setattr(self, key, save_state[key])

        logging.getLogger().info(f"Loaded from {ckpt_path}")

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    # def update_generalization_metrics(self, split, epoch, batch, logits, losses, model_out=None, compute_grads=True):
    #     # Cross-Entropy
    #     self.loss_visualizer.update(split, 'Running Loss', losses.mean().item())
    #     # if self.option.enable_groupwise_metrics:
    #     self.update_groupwise_values(split, 'Running Loss', losses, batch)
    def update_generalization_metrics(self, split, batch, losses):
        # Cross-Entropy
        self.loss_visualizer.update(split, 'Running Loss', losses.mean().item())
        # if self.option.enable_groupwise_metrics:
        self.update_groupwise_values(split, 'Running Loss', losses, batch)

    def _after_one_epoch(self, epoch, test_loaders, force_test=False):
        if epoch % self.option.save_model_every == 0:
            self.save(os.path.join(self.option.expt_dir, f'ckpt_epoch_{epoch}.pt'))

        if (self.option.test_epochs is not None and epoch in self.option.test_epochs) or (
                self.option.test_every is not None and epoch % self.option.test_every == 0) or force_test:
            for test_key in test_loaders:
                self.test(epoch, test_key, test_loaders[test_key])

    def after_all_epochs(self):
        """
        Saves all the metrics
        :return:
        """
        metrics_fname = os.path.join(self.option.expt_dir, 'metrics.json')
        json.dump(self.metrics, open(metrics_fname, 'w'), indent=4, sort_keys=True)

    def gather_gt_scores(self, logits, y):
        return logits.gather(1, y.view(-1, 1))

    def test_default(self, epoch, data_key, data_loader, model=None, model_key="Main"):
        logging.getLogger().info(f"\nEpoch {epoch}: Testing with data split: {data_key} model: {model_key}")
        if model is None:
            model = self.model
        self._mode_setting(is_train=False)

        ################################################################################################################
        # Initialize the metrics holders
        ################################################################################################################
        accuracy_metric = Accuracy(self.option.num_classes)
        group_type_to_accuracy_metric = {}
        losses = torch.ones(self.max_dataset_ixs[data_key] + 1).float() * -1000
        logits = torch.ones(self.max_dataset_ixs[data_key] + 1, self.option.num_classes).float() * -1000
        group_type_to_names = {}
        gt_labels = torch.ones(self.max_dataset_ixs[data_key] + 1).long() * -1000
        chart_name = f'{data_key}_{model_key}'

        ################################################################################################################
        # Now go through the data items, while computing the metrics and storing the predictions
        ################################################################################################################
        for i, batch in enumerate(data_loader):
            # Do a forward pass
            batch = self.prepare_batch(batch)
            labels = batch['y']
            out = self.forward_model(model, batch)
            batch_losses = self.compute_loss(out, torch.squeeze(labels))
            pred_ys = out['logits'].max(1)[1].detach().cpu().numpy()
            gt_ys = batch['y'].squeeze().cpu().numpy()

            accuracy_metric.update(pred_ys, gt_ys)

            ################################################################################################################
            # Initialize group details
            ################################################################################################################

            for key in batch.keys():
                if 'group' in key and 'name' in key:
                    if group_type_to_names is None or key not in group_type_to_names:
                        group_type_to_names[key] = np.asarray(
                            ['NoneNoneNoneNoneNoneNone'] * (self.max_dataset_ixs[
                                                                data_key] + 1))  # Hackish way to gather the space required for storing group names

                    if key not in group_type_to_accuracy_metric:
                        group_type_to_accuracy_metric[key] = GroupWiseAccuracy()
                    group_type_to_accuracy_metric[key].update(pred_ys, gt_ys, batch[key])
                    group_type_to_names[key][batch['dataset_ix']] = batch[key]

            ################################################################################################################
            # Store the results
            ################################################################################################################
            logits[batch['dataset_ix']] = out['logits'].detach().cpu()
            losses[batch['dataset_ix']] = batch_losses.detach().cpu()
            gt_labels[batch['dataset_ix']] = batch['y'].squeeze().cpu()
            self.loss_visualizer.update(chart_name, f'{model_key} Loss', batch_losses.detach().mean().item())

        curr_metric_entry = {}

        ################################################################################################################
        # Compute per-group metrics
        ################################################################################################################
        grp_type_to_mean_per_grp_accuracy = {}
        for grp_type in group_type_to_accuracy_metric:
            per_group_accuracy = group_type_to_accuracy_metric[grp_type].get_per_group_accuracy()
            grp_sum, n_grps = 0, 0
            for group in per_group_accuracy:
                self.accuracy_visualizer.update(chart_name, f'{group}', per_group_accuracy[group] * 100)
                grp_sum += per_group_accuracy[group] * 100
                n_grps += 1
            grp_type_to_mean_per_grp_accuracy[grp_type] = grp_sum / n_grps
            self.accuracy_visualizer.update(chart_name, f'Mean Per {grp_type}',
                                            grp_type_to_mean_per_grp_accuracy[grp_type])

        ################################################################################################################
        # Update unnormalized, per class and per group accuracies
        ################################################################################################################
        self.accuracy_visualizer.update(chart_name, f'{model_key} Accuracy', accuracy_metric.get_accuracy() * 100)
        self.accuracy_visualizer.update(chart_name, f'{model_key} MPA',
                                        accuracy_metric.get_mean_per_class_accuracy() * 100)
        self.accuracy_visualizer.log(epoch, chart_name)
        self.loss_visualizer.log(epoch, chart_name)
        logging.getLogger().info(
            "Mean per groups " + json.dumps(grp_type_to_mean_per_grp_accuracy, indent=4, sort_keys=True))
        save_file = os.path.join(self.option.expt_dir, f'preds_{chart_name}_epoch_{epoch}.pt')

        ################################################################################################################
        # Save the predictions
        ################################################################################################################
        if self.option.save_predictions_every is not None and epoch % self.option.save_predictions_every == 0:
            torch.save({
                'logits': logits,
                'losses': losses,
                'gt_labels': gt_labels,
                'group_type_to_names': group_type_to_names,
                'accuracy_metrics': self.accuracy_visualizer.metrics
            }, save_file)
            logging.getLogger().info(f"Saved to {save_file}")

        ################################################################################################################
        # Plot everything
        ################################################################################################################
        self.accuracy_visualizer.accumulate_plot_and_reset(epoch)

        ################################################################################################################
        # Update metrics
        ################################################################################################################
        if model_key not in self.metrics:
            # logging.getLogger().info(f"{model_key} not found in metrics")
            self.metrics[model_key] = {}
        if epoch not in self.metrics[model_key]:
            self.metrics[model_key][epoch] = {}
        curr_metric_entry['accuracy'] = accuracy_metric.get_accuracy() * 100
        curr_metric_entry['MPA'] = accuracy_metric.get_mean_per_class_accuracy() * 100

        self.metrics[model_key][epoch][data_key] = curr_metric_entry
        for grp_type in group_type_to_accuracy_metric:
            self.metrics[model_key][epoch][data_key + f'_{grp_type}'] = group_type_to_accuracy_metric[
                grp_type].get_per_group_accuracy()
            self.metrics[model_key][epoch][data_key + f'_{grp_type}_counts'] = group_type_to_accuracy_metric[
                grp_type].group_wise_total

    def test_gqa(self, epoch, data_key, data_loader, model=None, model_key="Main"):
        logging.getLogger().info(f"\nEpoch {epoch}: Testing with data split: {data_key} model: {model_key}")
        if model is None:
            model = self.model
        self._mode_setting(is_train=False)

        ################################################################################################################
        # Gather dataset details
        ################################################################################################################

        dataset = data_loader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        ix_to_ans = dataset.ix_to_ans
        ques_file_path = dataset.ques_file_path

        ################################################################################################################
        # Initialize variables that store the results
        ################################################################################################################
        predictions = {}
        losses = torch.ones(self.max_dataset_ixs[data_key] + 1).float() * -1000
        logits = torch.ones(self.max_dataset_ixs[data_key] + 1, self.option.num_classes).float() * -1000
        group_names = np.asarray(
            ['NoneNoneNoneNoneNoneNoneNoneNoneNoneNoneNoneNone'] * (
                    self.max_dataset_ixs[data_key] + 1))  # Hackish way to gather space to store group names
        gt_labels = torch.ones(self.max_dataset_ixs[data_key] + 1).long() * -1000
        chart_name = f'{data_key}_{model_key}'

        ################################################################################################################
        # Inference
        ################################################################################################################
        for i, batch in enumerate(data_loader):
            batch = self.prepare_batch(batch)
            labels = batch['y']
            self.optim.zero_grad()
            out = self.forward_model(model, batch)
            batch_losses = self.loss(out['logits'], torch.squeeze(labels))
            pred_ys = out['logits'].max(1)[1].detach().cpu().numpy()
            gt_ys = batch['y'].squeeze().cpu().numpy()

            for qid, pred_y in zip(batch['question_id'], pred_ys):
                pred_ans = ix_to_ans[str(pred_y)]
                predictions[qid] = pred_ans

            logits[batch['dataset_ix']] = out['logits'].detach().cpu()
            losses[batch['dataset_ix']] = batch_losses.detach().cpu()
            gt_labels[batch['dataset_ix']] = batch['y'].squeeze().cpu()
            group_names[batch['dataset_ix']] = batch['group_name']

            self.loss_visualizer.update(chart_name, f'{model_key} Loss', batch_losses.detach().mean().item())

        curr_metric_entry = {}

        self.loss_visualizer.log(epoch, chart_name)

        ################################################################################################################
        # Save the predictions
        ################################################################################################################
        save_file = os.path.join(self.option.expt_dir, f'preds_{chart_name}_epoch_{epoch}.pt')
        if epoch % self.option.save_model_every == 0:
            torch.save({
                'logits': logits,
                'losses': losses,
                'gt_labels': gt_labels,
                'group_names': group_names
            }, save_file)

        ans_preds_file = os.path.join(self.option.expt_dir, f'ans_preds_{chart_name}_epoch_{epoch}.json')
        json.dump(predictions, open(ans_preds_file, 'w'))
        if 'train' in data_key.lower():
            strict = False
        else:
            strict = True

        ################################################################################################################
        # Do GQA-style evaluation
        ################################################################################################################
        gqa_eval = GQAEval(predictions, ques_file_path, choices_path=None,
                           EVAL_CONSISTENCY=False, strict=strict)
        gqa_acc = gqa_eval.get_acc_result()['accuracy']
        result_string, detail_result_string = gqa_eval.get_str_result()
        for result_string_ in result_string:
            logging.getLogger().info(result_string_)
