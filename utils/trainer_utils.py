# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import torch
from torch import optim
from torch.optim import *


def save_option(option):
    if not os.path.exists(option.save_dir + '/' + option.expt_name):
        os.makedirs(option.save_dir + '/' + option.expt_name)
    option_path = os.path.join(option.save_dir, option.expt_name, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True,
                  default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")
    logging.getLogger().info(json.dumps(option.__dict__, indent=4, sort_keys=True,
                                        default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>"))


def initialize_logger(expt_dir):
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)

    null_handler = logging.NullHandler()
    null_handler.setLevel(logging.DEBUG)
    logging.getLogger("tornado.access").addHandler(null_handler)
    logging.getLogger("tornado.access").propagate = False

    file_handler = logging.FileHandler(os.path.join(expt_dir, 'log.txt'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class Timer(object):
    def __init__(self, logger, epochs, last_step=0):
        self.logger = logger
        self.epochs = epochs
        self.step = last_step

        curr_time = time.time()
        self.start = curr_time
        self.last = curr_time

    def __call__(self):
        curr_time = time.time()
        self.step += 1

        duration = curr_time - self.last
        remaining = (self.epochs - self.step) * (curr_time - self.start) / self.step / 3600
        msg = 'TIMER, duration(s)|remaining(h), %f, %f' % (duration, remaining)

        self.last = curr_time


def get_dir(file_path):
    return '/'.join(file_path.split('/')[:-1])


class GradMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mult(x, const):
    return GradMult.apply(x, const)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()  # * 0.1


def grad_reverse(x):
    return GradReverse.apply(x)


def create_optimizer(optimizer_name, named_params, lr, weight_decay=0, momentum=0.9, freeze_layers=None,
                     custom_lr_config=None):
    """
    Builds the optimizer of given name, adding only the provided named_params
    Supports freezing layers too

    For the experiments, we did not use freeze_layers or custom_lr_config. So, this code can be simplified a lot now.
    :return:
    """
    if weight_decay is None:
        weight_decay = 0

    def should_be_added(layer_name):
        ret = True
        if freeze_layers is None:
            return ret
        for freeze_layer in freeze_layers:
            if layer_name.startswith(freeze_layer):
                ret = False
        return ret

    filt_params = []
    for name, param in named_params:
        param_dict = None
        if should_be_added(name):
            if custom_lr_config is not None:
                for custom_lr_name in custom_lr_config:
                    if name.startswith(custom_lr_name):
                        param_dict = {'params': param, 'lr': custom_lr_config[custom_lr_name]}
            if param_dict is None:
                param_dict = {'params': param, 'lr': lr}
            filt_params.append(param_dict)
            # logging.getLogger().info(f"Adding param: {name}")
        else:
            param.requires_grad = False  # for efficiency
            logging.getLogger().info(f"Freezing param: {name}")

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(filt_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = eval(optimizer_name)(filt_params, lr=lr, weight_decay=weight_decay)
    return optimizer


def clip_grad_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == 'inf':
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm
