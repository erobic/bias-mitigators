import copy
import os


def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)


def biased_mnist_experiments(option, run):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA
    orig_option = copy.deepcopy(option)

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    for bias_variables in [['digit_color']]:
        for p_bias in [0.97]:
            option = copy.deepcopy(orig_option)
            run_expt(option, run, p_bias, bias_variables, '')


def biased_mnist_experiments_lr_wd(option, run):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA
    orig_option = copy.deepcopy(option)

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    for bias_variables in [['digit_color']]:
        for p_bias in [0.95]:
            for lr in [1e-3, 1e-4, 1e-5]:
                for wd in [0, 1e-3, 0.1]:
                    option = copy.deepcopy(orig_option)
                    option.lr = lr
                    option.weight_decay = wd
                    run_expt(option, run, p_bias, bias_variables, f'lr_{lr}_wd_{wd}')


def biased_mnist_experiments_p_bias(option, run):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA
    orig_option = copy.deepcopy(option)

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    for bias_variables in [['digit_color']]:
        for p_bias in [0.9, 0.93, 0.95, 0.97, 0.99, 1.0]:
            run_expt(orig_option, run, p_bias, bias_variables)


def biased_mnist_individual_variables(option, run_fn):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA
    orig_option = copy.deepcopy(option)

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    # 'digit_color',
    for bias_variable in ['digit_position', 'texture', 'texture_color', 'letter', 'letter_color', 'digit_color']:
        run_expt(orig_option, run_fn, 0.97, [bias_variable], '')


def biased_mnist_experiments_hierarchical(option, run_fn):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA
    orig_option = copy.deepcopy(option)

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    all_vars = []
    running_vars = []
    for bias_variable in ['digit_color', 'digit_position', 'texture', 'texture_color', 'letter', 'letter_color']:
        running_vars.append(bias_variable)
        all_vars.append(copy.deepcopy(running_vars))
    all_vars = list(reversed(all_vars))

    for vars in all_vars:
        run_expt(orig_option, run_fn, 0.97, vars, '')


def run_expt(orig_option, run_fn, p_bias, bias_variables, expt_name=''):
    option = copy.deepcopy(orig_option)
    option.do_not_print_by_long_group_name = True
    option.dataset_name = 'biased_mnist'
    option.data_dir = option.root_dir + f"/{option.dataset_name}"
    option.bias_split_name = 'full_v1'
    option.trainval_sub_dir = option.bias_split_name + "_" + str(p_bias)
    option.target_name = 'digit'
    option.bias_variables = bias_variables
    option.bias_variable_name = 'group_ix'  # For RUBi

    option.bias_model_hid_dims = 64  # RUBi
    option.bias_predictor_hid_dims = 64  # LNL
    option.num_classes = 10

    # Assumption: num_groups, bias_variable_dims and num_bias_classes will be set dynamically when creating the dataset

    # Optimizer + Model
    set_if_null(option, 'optimizer_name', 'Adam')
    set_if_null(option, 'batch_size', 128)
    set_if_null(option, 'epochs', 50)
    set_if_null(option, 'model_name', 'ResNet10')

    if option.trainer_name in ['GroupDROTrainer', 'GroupUpweightingTrainer']:
        if 'digit' not in option.bias_variables:
            option.bias_variables.append('digit')
    option.bias_variables_str = '+'.join(option.bias_variables)
    # Configure name of the experiments and directory to save the results
    if option.save_dir is None:
        option.save_dir = os.path.join(option.root_dir, option.project_name, option.trainval_sub_dir,
                                       f'target_{option.target_name}_bias_{option.bias_variables_str}',
                                       option.trainer_name)
    if option.trainer_name == 'IRMv1Trainer':
        option.save_dir += f'_grad_wt_{option.grad_penalty_weight}'
    if option.expt_name is None and expt_name is not None:
        option.expt_name = expt_name

    # Method-specific configurations
    if option.trainer_name == 'GroupDROTrainer':
        option.balanced_sampling_attributes = ['group_ix']  # Perform balanced sampling
        option.group_by = 'group_ix'  # Groups for GDRO
        option.key_to_group_by = 'group_name'  # Used to find readable group names

    if option.trainer_name == 'RUBiTrainer':
        set_if_null(option, 'bias_model_hid_dims', 64)
        # We experimented with MLP2 and MLP3 too, but MLP1 worked the best in the preliminary experiments
        set_if_null(option, 'bias_model_name', 'MLP1')
        set_if_null(option, 'bias_variable_type', 'categorical')

    if option.trainer_name == 'LNLTrainer':
        # We experimented with MLP2 and MLP3 too, but MLP1 had worked the best in the preliminary experiments
        feature_dims = get_feature_dims(option.model_name, option.num_classes)
        set_if_null(option, 'bias_predictor_name', 'MLP1')
        set_if_null(option, 'bias_predictor_in_layer', 'model.pooled2')
        option.bias_predictor_in_dims = feature_dims[option.bias_predictor_in_layer]
        option.bias_predictor_hid_dims = feature_dims[option.bias_predictor_in_layer]

    if option.trainer_name == 'IRMv1Trainer':
        set_if_null(option, 'num_envs_per_batch', 4)

    # Test epochs
    option.test_epochs = [e for e in
                          range(40,
                                51)]  # Due to instability, we average accuracies over the last 10 epochs
    save_every = 25
    option.test_every = save_every  # We further test every 10 epochs
    option.save_predictions_every = save_every
    option.save_model_every = save_every

    run_fn(option)


def get_feature_dims(model_name, num_classes):
    if 'ResNet10' in model_name:
        return {
            'model.conv1': 32,
            'model.pooled_conv1': 32,
            'model.layer1.1.conv2': 32,
            'model.pooled1': 32,
            'model.layer2.1.conv2': 64,
            'model.layer2_flattened': 64 * 28 * 28,
            'model.pooled2': 64,
            'model.layer3.1.conv2': 128,
            'model.pooled3': 128,
            'model.layer4.1.conv2': 256,
            'model.pooled4': 256,
            'model.fc': 10,
            'logits': num_classes
        }
