import numpy as np
import logging
import torch


class Accuracy():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def update(self, pred_ys, gt_ys):
        for pred_y, gt_y in zip(pred_ys, gt_ys):
            if pred_y == gt_y:
                self.correct[pred_y] += 1
            self.total[gt_y] += 1

    def get_accuracy(self):
        return self.correct.sum() / self.total.sum()

    def get_per_class_accuracy(self):
        correct = np.asarray(self.correct)
        total = np.asarray(self.total)
        for tix, (c, t) in enumerate(zip(correct, total)):
            if t == 0:
                correct[tix] = 1
                total[tix] = 1
        return correct / total

    def get_mean_per_class_accuracy(self):
        return self.get_per_class_accuracy().mean()

    def reset(self):
        self.correct = np.zeros((self.num_classes))
        self.total = np.zeros((self.num_classes))


class GroupWiseAccuracy():
    def __init__(self):
        self.reset()

    def update(self, pred_ys, gt_ys, group_names):
        for pred_y, gt_y, group_name in zip(pred_ys, gt_ys, group_names):
            group_name = str(group_name)
            if group_name not in self.group_wise_total:
                self.group_wise_total[group_name] = 0
                self.group_wise_correct[group_name] = 0
            if pred_y == gt_y:
                self.group_wise_correct[group_name] += 1
            self.group_wise_total[group_name] += 1

    def get_per_group_accuracy(self):
        per_group_accuracy = {}
        for group_name in self.group_wise_correct:
            per_group_accuracy[group_name] = self.group_wise_correct[group_name] / self.group_wise_total[group_name]
        return per_group_accuracy

    def get_mean_per_group_accuracy(self):
        total_acc, total_num = 0, 0
        per_group_accuracy = self.get_per_group_accuracy()
        for group_name in per_group_accuracy:
            total_acc += per_group_accuracy[group_name]
            total_num += 1
        return total_acc / total_num

    def reset(self):
        self.group_wise_correct = {}
        self.group_wise_total = {}

    def log(self, prefix=''):
        log_str = prefix
        per_group_accuracy = self.get_per_group_accuracy()
        group_names = sorted([k for k in per_group_accuracy.keys()])
        accuracies = ""
        for group_name in group_names:
            log_str += '%s: %.2f%% ' % (group_name, per_group_accuracy[group_name] * 100)
            accuracies += '%.2f%%, ' % (per_group_accuracy[group_name] * 100)
        log_str += ' MPG: %.2f%%' % (self.get_mean_per_group_accuracy() * 100)
        # logging.getLogger().info(log_str)
        logging.getLogger().info(f"Group names {group_names}")
        logging.getLogger().info(f"Accuracies {accuracies}")


class ModuleStatsComputer:
    """Stores sensitivities, GT classes and predicted classes"""

    def __init__(self, num_modules, num_classes):
        self.reset()
        self.num_modules = num_modules
        self.num_classes = num_classes

    def reset(self):
        self.sensitivities = []
        self.gt_class_ixs, self.pred_class_ixs = [], []
        self.group_names = []

    def update(self, sensitivities, gt_class_ixs, pred_class_ixs, group_names):
        self.sensitivities += sensitivities
        self.gt_class_ixs += gt_class_ixs
        self.pred_class_ixs += pred_class_ixs
        self.group_names += [str(gn) for gn in group_names]

    def log(self):
        sensitivities = np.asarray(self.sensitivities)
        gt_class_ixs = np.asarray(self.gt_class_ixs)
        pred_class_ixs = np.asarray(self.pred_class_ixs)
        most_sensitive_ixs = np.argmax(sensitivities, axis=1)
        group_names = np.asarray(self.group_names)

        most_sensitive_counts = {ix: len(np.nonzero(most_sensitive_ixs == ix)[0])
                                 for ix in range(0, self.num_modules)}
        most_sens_n_correct = {}
        accuracy_when_most_sensitive = {}
        group_distribution = {}
        overall_metrics = {}
        for module_ix in range(self.num_modules):
            most_sens_n_correct[module_ix] = len(np.intersect1d(np.nonzero(most_sensitive_ixs == module_ix)[0],
                                                                np.nonzero(gt_class_ixs == pred_class_ixs)[0]))
            accuracy_when_most_sensitive[module_ix] = 100 * most_sens_n_correct[module_ix] / max(most_sensitive_counts[
                                                                                                     module_ix], 1)
            if module_ix not in group_distribution:
                group_distribution[module_ix] = {}
            for gn in group_names[np.nonzero(most_sensitive_ixs == module_ix)[0]]:
                if gn not in group_distribution[module_ix]:
                    group_distribution[module_ix][gn] = 0
                group_distribution[module_ix][gn] += 1

        for module_ix in range(self.num_modules):
            if int(most_sensitive_counts[module_ix]) > 0:
                overall_metrics[module_ix] = {
                    'most_sensitive_count': int(most_sensitive_counts[module_ix]),
                    'accuracy_when_most_sensitive': '%.2f%%' % (float(accuracy_when_most_sensitive[module_ix])),
                    'group_distribution': group_distribution[module_ix]
                }


class GradientTracker():
    def __init__(self, num_samples, num_epochs):
        # l2_norm = torch.norm(gradients.detach(), p=2, dim=1)
        # abs = torch.sum(torch.abs(gradients.detach()), dim=1)

        self.l2_norms = torch.zeros((num_samples, num_epochs))
        self.abs_vals = torch.zeros((num_samples, num_epochs))
        self.groups = np.asarray(['NoneNoneNoneNoneNoneNone'] * (num_samples))
        self.unq_groups = {}

    def update(self, epoch, dataset_ixs, gradients, groups):
        self.epoch = epoch
        # self.l2_norms[dataset_ixs, epoch - 1] = torch.norm(gradients.detach().flatten(1), p=2, dim=1).cpu()
        # self.abs_vals[dataset_ixs, epoch - 1] = torch.sum(torch.abs(gradients.detach().flatten(1)), dim=1).cpu()
        if epoch == 1:
            self.groups[dataset_ixs] = groups
            for g in groups:
                if g not in self.unq_groups:
                    self.unq_groups[g] = g

    def get_groupwise_values(self):
        # return mean, variance, normalized variance, mean of running variance?
        # How does it evolve over time
        values = {}
        for g in self.unq_groups:
            if g not in values:
                values[g] = {}
            grp_ixs = np.nonzero(self.groups == g)[0]
            l2_norms = self.l2_norms[grp_ixs, self.epoch - 1]
            values[g]['mean_of_l2_norms'] = torch.mean(l2_norms)
            values[g]['variance_of_l2_norms'] = torch.std(l2_norms) ** 2

            squared_l2_norms = l2_norms ** 2
            values[g]['mean_of_squared_l2_norms'] = torch.mean(squared_l2_norms)
            values[g]['variance_of_squared_l2_norms'] = torch.std(squared_l2_norms) ** 2

            abs = self.abs_vals[grp_ixs, self.epoch - 1]
            values[g]['mean_of_abs'] = torch.mean(abs)
            values[g]['variance_of_abs'] = torch.std(abs) ** 2

        return values


class PredictionChangeTracker():
    def __init__(self, num_samples, num_epochs):
        self.preds = torch.zeros((num_samples, num_epochs))
        self.num_pred_changes = torch.zeros((num_samples))
        self.groups = np.asarray(['NoneNoneNoneNoneNoneNone'] * (num_samples))
        self.unq_groups = {}

    def update(self, dataset_ixs, epoch, logits, labels, groups):
        self.epoch = epoch
        self.preds[dataset_ixs, epoch - 1] = torch.argmax(logits.cpu(), dim=1).float()
        if epoch > 1:
            pred_change_mask = torch.where(self.preds[dataset_ixs, epoch - 2] != self.preds[dataset_ixs, epoch - 1],
                                           torch.ones_like(dataset_ixs),
                                           torch.zeros_like(dataset_ixs))
            self.groups[dataset_ixs] = groups
            for g in groups:
                if g not in self.unq_groups:
                    self.unq_groups[g] = g
            self.num_pred_changes[dataset_ixs] += pred_change_mask

    def get_values(self):
        group_change_pct = {}
        max_num_changes = {}
        mean_num_changes = {}
        for g in self.unq_groups:
            grp_ixs = np.nonzero(self.groups == g)[0]
            num_changes = self.num_pred_changes[grp_ixs].sum()
            grp_len = len(grp_ixs)
            group_change_pct[g] = (num_changes / (grp_len * (self.epoch - 1))) * 100
            max_num_changes[g] = self.num_pred_changes[grp_ixs].max().item()

        return {
            'group_change_percent': group_change_pct,
            'max_num_changes': max_num_changes,
        }
