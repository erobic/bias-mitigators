import os
from analysis.biased_mnist.common import *


def collect_table1():
    dataset_name = 'full_v1_0.97'
    implicit_methods = ['BaseTrainer', 'LffTrainer', 'SpectralDecouplingTrainer']
    method_to_acc = {}
    for method in implicit_methods:
        acc = read_test_accuracy(os.path.join(method, 'target_digit_bias_digit_color'))
        method_to_acc[method] = acc



if __name__ == "__main__":
    collect_table1()
