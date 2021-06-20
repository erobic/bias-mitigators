import os
import json


def main():
    for trainer_name in ['BaseTrainer', 'GroupUpweightingTrainer', 'GroupDROTrainer',
                         'LNLTrainer', 'IRMv1Trainer', 'RUBiTrainer',
                         'LffTrainer', 'SpectralDecouplingTrainer']:
        root = '/hdd/robik/Bias-Mitigators/full_v1_0.9/target_digit_bias_digit_color'
        best_lr, best_wd = 0, 0
        best_acc = 0
        for lr in [0.001, 0.0001, 1e-05]:
            for wd in [0, 0.001, 0.1]:
                expt_dir = os.path.join(root, trainer_name, f'lr_{lr}_wd_{wd}')
                with open(os.path.join(expt_dir, 'metrics.json')) as f:
                    acc = json.load(f)['Main']['50']['Test']['accuracy']
                    if acc > best_acc:
                        best_lr = lr
                        best_wd = wd
                        best_acc = acc
        print(f"{trainer_name},{best_lr},{best_wd}" + ",%.2f" % best_acc)


if __name__ == "__main__":
    main()
