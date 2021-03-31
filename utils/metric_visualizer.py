import numpy as np
import logging
import torch
# import neptune
import json


class MetricVisualizer():
    def __init__(self, env_name='main', divide_windows_by='split', is_one_time_metric=True):
        self.env_name = env_name
        self.plots = {}
        self.reset()
        self.divide_windows_by = divide_windows_by
        self.use_markers = False
        self.is_one_time_metric = is_one_time_metric

    def update_multiple(self, split, dict):
        for name, value in zip(dict.keys(), dict.values()):
            self.update(split, name, value)

    def update(self, split, name, value):
        if split not in self.metrics:
            self.metrics[split] = {}
            self.metric_cnts[split] = {}
        if name not in self.metrics[split]:
            if self.is_one_time_metric:
                self.metrics[split][name] = 0
            else:
                self.metrics[split][name] = []
            self.metric_cnts[split][name] = 0

        if self.is_one_time_metric:
            self.metrics[split][name] = value
            self.metric_cnts[split][name] = 1
        else:
            if isinstance(value, list):
                for val in value:
                    self.metrics[split][name].append(val)
                    self.metric_cnts[split][name] += 1
            else:
                self.metrics[split][name].append(value)
                self.metric_cnts[split][name] += 1

    def reset(self):
        self.metrics = {}
        self.metric_cnts = {}

    def compute_and_save_std_dev(self, split, name):
        if split + " Std Dev" not in self.metrics:
            self.metrics[split + " Std Dev"] = {}
            self.metrics[split + " Variance"] = {}

        self.metrics[split + " Std Dev"][name] = torch.std(torch.Tensor(self.metrics[split][name]))
        self.metrics[split + " Variance"][name] = self.metrics[split + " Std Dev"][name] ** 2

    def log(self, epoch, split, avg=True):
        log_str = f"Split: {split}"
        if epoch is not None:
            log_str += f', Epoch: {epoch}'
        name_values = {}
        for ix, name in enumerate(self.metrics[split]):
            if self.is_one_time_metric:
                val = self.metrics[split][name]
            else:
                val = sum(self.metrics[split][name])
                if avg:
                    val = val / (self.metric_cnts[split][name] + int(self.metric_cnts[split][name] == 0))
            metric_format = self.get_metric_format()
            if 'cnt' in name.lower():
                metric_format = '%d'

            # log_str += (", %s: " + metric_format) % (name, val)
            # neptune.log_metric(log_name=split + " " + name, x=epoch, y=val)
            name_values[name] = val
        if len(name_values) <= 1:
            logging.getLogger().info(name_values)
        else:
            logging.getLogger().info(json.dumps(name_values, sort_keys=True, indent=4))
        # logging.getLogger().info(sorted(list(name_values.keys())))
        # vals = [float(name_values[k]) for k in sorted(list(name_values.keys()))]
        # logging.getLogger().info(vals)

        # logging.getLogger().info(log_str)

    def get_metric_format(self):
        return "%.4f"

    def accumulate_plot_and_reset(self, epoch, xlabel='Epochs', avg=True):
        x = epoch
        for split in self.metrics:
            for name in self.metrics[split]:
                if self.is_one_time_metric:
                    y = self.metrics[split][name]
                else:
                    try:
                        y = sum(self.metrics[split][name])
                        if avg:
                            y = y / (self.metric_cnts[split][name] + int(self.metric_cnts[split][name] == 0))
                    except:
                        y = self.metrics[split][name]
                self.plot(split, name, x, y, xlabel)
        self.reset()

    def plot(self, split, name, x, y, xlabel='Epochs'):
        """
        Creates a separate chart for each loss
        :return:
        """
        if self.divide_windows_by == 'name':
            win_name = name
            line_name = split
        else:
            win_name = split
            line_name = name
        # if win_name not in self.plots:
        #     self.plots[win_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env_name,
        #                                          opts=dict(legend=[line_name], title=win_name, xlabel=xlabel,
        #                                                    ylabel=name, markers=self.use_markers))
        #
        # else:
        #     self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env_name, win=self.plots[win_name], name=line_name,
        #                   update='append')


class LossVisualizer(MetricVisualizer):
    def __init__(self, env_name='main', divide_windows_by='split'):
        super(LossVisualizer, self).__init__(env_name, divide_windows_by, is_one_time_metric=False)
        self.use_markers = False

    def get_metric_format(self):
        return "%.4f"


class AccuracyVisualizer(MetricVisualizer):
    def __init__(self, env_name='main', divide_windows_by='split'):
        super(AccuracyVisualizer, self).__init__(env_name, divide_windows_by, is_one_time_metric=True)
        self.use_markers = True

    def get_metric_format(self):
        return "%.2f%%"


class CountVisualizer(MetricVisualizer):
    def __init__(self, env_name='main', divide_windows_by='name'):
        super(CountVisualizer, self).__init__(env_name, divide_windows_by, is_one_time_metric=True)
        self.use_markers = False

    def get_metric_format(self):
        return "%d"

    def log(self, epoch, split):
        return super().log(epoch, split)

    def accumulate_plot_and_reset(self, epoch, xlabel='Epochs'):
        return super().accumulate_plot_and_reset(epoch, xlabel)
