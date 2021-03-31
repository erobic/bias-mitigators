# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--expt_name', required=False, help='Experiment will be saved under this name.')
parser.add_argument('--expt_type', type=str,
                    help='The experimental configuration to use e.g., all CelebA runs can use celebA_experiments value for this argument')
parser.add_argument('--dataset_name', required=False, help='Name of the dataset')
parser.add_argument('--root_dir', default=None)
parser.add_argument('--num_classes', type=int, help='Number of classes')
parser.add_argument('--batch_size', default=None, type=int, help='Mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay')
parser.add_argument('--epochs', type=int, help='Num of epochs')
parser.add_argument('--load_checkpoint', default=None, help='Checkpoint to resume from.')

parser.add_argument('--test_every', default=5, type=int, help='Interval to perform testing')
parser.add_argument('--test_epochs', nargs='+', type=int,
                    help='List of epochs to perform testing at. Is compatible with test_every argument -- both will be used')
parser.add_argument('--save_predictions_every', default=25, type=int, help='Interval to save predictions and metrics')
parser.add_argument('--save_model_every', default=25, type=int, help='Interval to save the model checkpoint')
parser.add_argument('--data_dir', required=False,
                    help='Directory where the dataset is stored. We usually assume this convention: {root_dir}/{dataset_name}')
parser.add_argument('--save_dir', required=False, help='Logs, Checkpoints, predictions and metrics will be saved here.')

parser.add_argument('--random_seed', type=int, help='Random seed', default=1)
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers in data loader')

parser.add_argument('--grad_clip', type=float, default=None,
                    help="Grad clip. This wasn't used for any of the methods for the comparison experiments.")
parser.add_argument('--trainer_name', type=str, help="Name of the method e.g., BaseTrainer or GroupDROTrainer")
parser.add_argument('--model_name', type=str,
                    help="Name of the main model. For two branch models e.g., RUBi, this refers to the name for the main branch.")
parser.add_argument('--bias_model_name', type=str,
                    help="For two/multi branch setups, this either predicts the bias variables or uses them as input.")
parser.add_argument('--optimizer_name', type=str, default=None, help="e.g., SGD, Adam")
parser.add_argument('--bias_proba', type=float, default=1.1, help='p_bias for BiasedMNIST')
parser.add_argument('--bias_var', type=float, default=0.02)
parser.add_argument('--dummy', action='store_true',
                    help="A flag used for debugging runs e.g., setting num_workers=0 to make debugging possible and using a smaller dataset size.")
parser.add_argument('--balanced_sampling_attributes', type=str, nargs='+', default=None,
                    help="List of attributes (as returned in a mini-batch) which should be used for balancing i.e., every unique combination of these attributes will have equal probability of being sampled."
                         "Useful for GroupDRO")
parser.add_argument('--balanced_sampling_gamma', type=float, default=1.0,
                    help="Exponentiation for inverse group probability. Higher values would oversample minority patterns a lot.")
parser.add_argument('--freeze_layers', default=None, nargs='+',
                    help="Can be used to freeze layers i.e., not used for optimization."
                         "When freezing, you need to disable batch norm and other model-specific settings yourself.")
parser.add_argument('--custom_lr_config', default=None, type=str, help="Unused (deprecated) argument.")

parser.add_argument('--grad_reverse_factor', type=float, default=-0.1,
                    help="Reversal parameter for adversarial debiasing e.g., learning not to learn (LNL). Use a negative value.")
parser.add_argument('--loss_type', type=str, default='CrossEntropyLoss')

# Arguments specific to GroupDROTrainer
parser.add_argument('--num_groups', type=int, help="Number of groups for grouping methods e.g., GroupDRO.")
parser.add_argument('--group_weight_step_size', type=float, default=0.01,
                    help="Learning rate to update group weights in GroupDRO.")
parser.add_argument('--group_mode', type=str,
                    help='Grouping mode e.g., unique_bias_value or majority_minority for BiasedMNIST. TODO: remove this.')
parser.add_argument('--bias_predictor_in_layer', type=str, default=None,
                    help="LNL predicts bias variables from this layer.")
parser.add_argument('--bias_predictor_name', type=str, default=None, help="Bias model name for LNL.")

parser.add_argument('--bias_variable_name', type=str, default=None,
                    help="Name of the bias variable used by explicit methods and also used to compute metrics.")
parser.add_argument('--target_name', type=str, default=None, help="Variable name to predict i.e., class variable.")
parser.add_argument('--group_by', type=str, default=None,
                    help="Dataset is grouped by this variable, usually set to group_ix.")
parser.add_argument('--key_to_group_by', type=str, default=None, help="This provides names for the groups.")

# Arguments specific to LffTrainer
parser.add_argument('--bias_loss_gamma', type=float, default=0.7, help="Loss gamma for LFF")
parser.add_argument('--bias_ema_gamma', type=float, default=0.7, help="EMA gamma for LFF")
parser.add_argument('--bias_model_hid_dims', type=int, help='Hidden dimensions for the bias model')

parser.add_argument('--entropy_loss_weight', type=float, default=0, help="Weight for entropy loss weight in LNL.")

parser.add_argument('--dataset_info', help="Used internally to set dataset specific attributes.")
parser.add_argument('--enable_groupwise_metrics', action='store_true')
parser.add_argument('--project_name', type=str, default='Bias-Mitigators', help="Results will be saved here.")

# Arguments specific to RunningFocalLossTrainer
parser.add_argument('--in_dims', type=int, default=None)
parser.add_argument('--hid_dims', type=int, default=None)
parser.add_argument('--grad_penalty_weight', type=float, default=1.0)
parser.add_argument('--expt_dir', type=str)
parser.add_argument('--bias_variable_type', type=str)

parser.add_argument('--spectral_decoupling_lambda', type=float)
parser.add_argument('--spectral_decoupling_lambdas', type=float, nargs='+')
parser.add_argument('--spectral_decoupling_gamma', type=float)
parser.add_argument('--spectral_decoupling_gammas', type=float, nargs='+')

parser.add_argument('--num_envs_per_batch', type=int,
                    help="Used by IRMv1. Each mini-batch will contain the specified number of environments.")


def get_option():
    option = parser.parse_args()
    option.cuda = True
    if option.dummy:
        option.num_workers = 0
    return option


# Used when bash files are not used
ROOT = '/hdd/user'
EXPT_ROOT = '/hdd/user/bias_mitigators'
