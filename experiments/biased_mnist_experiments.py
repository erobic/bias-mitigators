import os
# from main import get_option, run
import copy
import numpy as np
from datasets.biased_mnist_dataset import FactorToValues


def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)


def set_default_values(option):
    # Each unique value of a bias variable is considered to be a different group
    option.dataset_name = 'biased_mnist_v1'
    option.data_dir = option.root_dir + f"/{option.dataset_name}"
    option.num_classes = 10
    option.test_epochs = [15, 30]
    option.test_every = 15
    option.save_predictions_every = 30
    option.save_model_every = 30
    if option.trainer_name in ['GroupDROTrainer']:
        option.balanced_sampling_attributes = ['group_ix']
        option.group_by = 'group_ix'
        option.key_to_group_by = 'group_name'
    if option.expt_name is None:
        option.expt_name = f"lr_{option.lr}_wd_{option.weight_decay}"
    set_if_null(option, 'optimizer_name', 'Adam')
    set_if_null(option, 'batch_size', 128)
    set_if_null(option, 'epochs', 30)
    set_if_null(option, 'model_name', 'BiasedMNISTCNN')


def biased_mnist_experiments(option, run):
    orig_option = copy.deepcopy(option)
    for bias_variables in [['distractor']]:  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        for bias_proba in [0.7]:
            option = copy.deepcopy(orig_option)
            set_default_values(option)

            option.bias_proba = bias_proba
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            if option.group_mode == 'majority_minority':
                option.use_majority_minority_grouping = True
            else:
                option.group_mode = 'unique_bias_value'
                option.use_majority_minority_grouping = False

                if 'group' in option.trainer_name.lower():
                    option.bias_variables.append('label')  # used by biased_mnist dataset

            option.bias_variables_str = '+'.join(option.bias_variables)

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, f'AVID_rand_seed_{option.random_seed}',
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            run(option)


def biased_mnist_experiments_p_bias(option, run):
    # Each unique value of a bias variable is considered to be a different group
    orig_option = copy.deepcopy(option)
    for bias_variables in [['distractor']]:  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        for bias_proba in [0.1, 0.5, 0.6, 0.7, 0.8, 0.9]:  # 0.1, 0.5, 0.6, 0.7, 0.8, 0.9
            option = copy.deepcopy(orig_option)
            set_default_values(option)

            option.bias_proba = bias_proba
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            option.group_mode = 'unique_bias_value'
            option.use_majority_minority_grouping = False

            if 'group' in option.trainer_name.lower():
                option.bias_variables.append('label')  # used by biased_mnist dataset

            # option.bias_variable_name = '+'.join(bias_variables)
            option.bias_variables_str = '+'.join(option.bias_variables)

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, f'AVID_rand_seed_{option.random_seed}',
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            run(option)


def biased_mnist_experiments_individual_variables(option, run):
    # Each unique value of a bias variable is considered to be a different group
    orig_option = copy.deepcopy(option)
    ftv = FactorToValues().factor_to_values

    for fv in ftv.keys():  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        bias_variables = [fv]
        for bias_proba in [0.7]:  # 0.1, 0.5, 0.6, 0.7, 0.8, 0.9
            # for bias_proba in [0.7]:
            option = copy.deepcopy(orig_option)
            set_default_values(option)

            option.bias_proba = bias_proba
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            if option.group_mode == 'majority_minority':
                option.use_majority_minority_grouping = True

            else:
                option.group_mode = 'unique_bias_value'
                option.use_majority_minority_grouping = False

                if 'group' in option.trainer_name.lower():
                    option.bias_variables.append('label')  # used by biased_mnist dataset

            # option.bias_variable_name = '+'.join(bias_variables)
            option.bias_variables_str = '+'.join(option.bias_variables)

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, f'AVID_rand_seed_{option.random_seed}',
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            run(option)


def biased_mnist_experiments_all_variables(option, run):
    # Each unique value of a bias variable is considered to be a different group
    orig_option = copy.deepcopy(option)
    ftv = FactorToValues().factor_to_values
    bias_variables = list(ftv.keys())
    for bias_proba in [0.7]:  # 0.1, 0.5, 0.6, 0.7, 0.8, 0.9
        option = copy.deepcopy(orig_option)
        set_default_values(option)
        option.bias_proba = bias_proba
        # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
        option.bias_variables = copy.deepcopy(bias_variables)
        option.bias_variable_name = 'group_ix'  # used by rubi

        if option.group_mode == 'majority_minority':
            option.use_majority_minority_grouping = True
        else:
            option.group_mode = 'unique_bias_value'
            option.use_majority_minority_grouping = False

            if 'group' in option.trainer_name.lower():
                option.bias_variables.append('label')  # used by biased_mnist dataset

        option.bias_variables_str = '+'.join(option.bias_variables)
        if option.save_dir is None:
            option.save_dir = os.path.join(option.root_dir, option.project_name,
                                           option.dataset_name + f"_{option.bias_proba}",
                                           option.group_mode,
                                           option.bias_variables_str,
                                           option.trainer_name)
        run(option)


def biased_mnist_experiments_distractor(option, run):
    # Each unique value of a bias variable is considered to be a different group
    orig_option = copy.deepcopy(option)
    for bias_variables in [['distractor']]:  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        for bias_proba in [0.7]:  # 0.1, 0.5, 0.6, 0.7, 0.8, 0.9
            option = copy.deepcopy(orig_option)
            set_default_values(option)

            option.bias_proba = bias_proba
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            option.group_mode = 'unique_bias_value'
            option.use_majority_minority_grouping = False

            if 'group' in option.trainer_name.lower():
                option.bias_variables.append('label')  # used by biased_mnist dataset

            # option.bias_variable_name = '+'.join(bias_variables)
            option.bias_variables_str = '+'.join(option.bias_variables)

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, option.project_name,
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            run(option)


def biased_mnist_experiments_hierarchical_majority_minority_groups(option, run):
    orig_option = copy.deepcopy(option)
    composite_bias_variables = []
    for ix, curr_bias_variable in enumerate(
            ['distractor', 'texture', 'digit_color', 'bg_color', 'texture_color', 'distractor_color',
             'digit_cell_number']):
        composite_bias_variables.append(curr_bias_variable)
        for bias_proba in [0.7]:
            option = copy.deepcopy(orig_option)
            set_default_values(option)
            option.bias_proba = bias_proba
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(composite_bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            option.group_mode = 'majority_minority'
            option.use_majority_minority_grouping = True
            if 'group' in option.trainer_name.lower():
                option.bias_variables.append('label')  # used by biased_mnist dataset

            option.bias_variables_str = '+'.join(option.bias_variables)
            option.test_epochs = [15, 30]
            option.test_every = 15
            option.save_predictions_every = 30
            option.save_model_every = 30

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, option.project_name,
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            run(option)


def biased_mnist_experiments_hierarchical_unique_groups(option, run):
    orig_option = copy.deepcopy(option)
    composite_bias_variables = []
    for ix, curr_bias_variable in enumerate(
            ['distractor', 'texture', 'digit_color', 'bg_color', 'texture_color', 'distractor_color',
             'digit_cell_number']):
        composite_bias_variables.append(curr_bias_variable)
        for bias_proba in [0.7]:
            option = copy.deepcopy(orig_option)
            set_default_values(option)
            option.bias_proba = bias_proba

            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(composite_bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            option.group_mode = 'unique_bias_value'
            option.use_majority_minority_grouping = False
            if 'group' in option.trainer_name.lower():
                option.bias_variables.append('label')  # used by biased_mnist dataset

            option.bias_variables_str = '+'.join(option.bias_variables)
            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, f'AVID_rand_seed_{option.random_seed}',
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            run(option)


def biased_mnist_experiments_p_bias_lff(option, run):
    # Each unique value of a bias variable is considered to be a different group
    orig_option = copy.deepcopy(option)
    for bias_variables in [['distractor']]:  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        for bias_proba in [0.1, 0.5, 0.6, 0.7, 0.8, 0.9]:  # 0.1, 0.5, 0.6, 0.7, 0.8, 0.9
            option = copy.deepcopy(orig_option)
            set_default_values(option)
            option.bias_proba = bias_proba
            option.num_classes = 10
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            option.group_mode = 'unique_bias_value'
            option.use_majority_minority_grouping = False

            if 'group' in option.trainer_name.lower():
                option.bias_variables.append('label')  # used by biased_mnist dataset

            option.bias_variables_str = '+'.join(option.bias_variables)

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, f'AVID_rand_seed_{option.random_seed}',
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name)
            if option.expt_name is None:
                option.expt_name = f"lr_{option.lr}_wd_{option.weight_decay}_bias_loss_gamma_{option.bias_loss_gamma}"
            run(option)


def biased_mnist_p_bias_individual_variables(option, run):
    # Each unique value of a bias variable is considered to be a different group
    orig_option = copy.deepcopy(option)
    ftv = FactorToValues().factor_to_values

    for fv in ftv.keys():  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        bias_variables = [fv]
        option = copy.deepcopy(orig_option)
        set_default_values(option)
        option.bias_variables = copy.deepcopy(bias_variables)
        option.bias_variable_name = 'group_ix'  # used by rubi

        option.group_mode = 'unique_bias_value'
        option.use_majority_minority_grouping = False

        if 'group' in option.trainer_name.lower():
            option.bias_variables.append('label')  # used by biased_mnist dataset

        option.bias_variables_str = '+'.join(option.bias_variables)

        if option.save_dir is None:
            option.save_dir = os.path.join(option.root_dir, f'AVID_rand_seed_{option.random_seed}',
                                           option.dataset_name + f"_{option.bias_proba}",
                                           option.group_mode,
                                           option.bias_variables_str,
                                           option.trainer_name)
        run(option)


def biased_mnist_experiments_coordconv(option, run):
    # Each unique value of a bias variable is considered to be a different group
    option.dataset_name = 'biased_mnist_v1'
    option.data_dir = option.root_dir + f"/{option.dataset_name}"

    orig_option = copy.deepcopy(option)
    for bias_variables in [['digit_cell_number']]:  # ['fg_color', 'bg_color'], ['fg_color', 'bg_color', 'texture']
        for bias_proba in [0.7]:
            option = copy.deepcopy(orig_option)
            set_default_values(option)
            option.model_name = 'BiasedMNISTCoordConv'
            option.bias_proba = bias_proba
            # Won't be used by implicit methods e.g., BaseTrainer, LffTrainer, SpectralDecouplingTrainer
            option.bias_variables = copy.deepcopy(bias_variables)
            option.bias_variable_name = 'group_ix'  # used by rubi

            if option.group_mode == 'majority_minority':
                option.use_majority_minority_grouping = True

            else:
                option.group_mode = 'unique_bias_value'
                option.use_majority_minority_grouping = False

                if 'group' in option.trainer_name.lower():
                    option.bias_variables.append('label')  # used by biased_mnist dataset

            option.bias_variables_str = '+'.join(option.bias_variables)

            if option.save_dir is None:
                option.save_dir = os.path.join(option.root_dir, option.project_name,
                                               option.dataset_name + f"_{option.bias_proba}",
                                               option.group_mode,
                                               option.bias_variables_str,
                                               option.trainer_name + "_coord_conv")
            run(option)
