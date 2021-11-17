import os
import json

RESULTS_PATH = '/hdd/robik/Bias-Mitigators'


def read_metrics(path):
    return json.load(open(os.path.join(path, 'metrics.json')))


def read_test_accuracy(path, split='Test', epoch='50'):
    metrics = read_metrics(path)
    return metrics[epoch][split]['accuracy']
