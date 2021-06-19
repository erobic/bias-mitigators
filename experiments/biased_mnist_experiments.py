import os
import copy


def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)


def biased_mnist_experiments(option, run):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA
    orig_option = copy.deepcopy(option)

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    for lr in [1e-3, 1e-4, 1e-5]:
        for wd in [0, 0.1, 1e-3]:
            for bias_variables in [['digit_color']]:
                for p_bias in [0.9]:
                    option = copy.deepcopy(orig_option)
                    option.dataset_name = 'biased_mnist'
                    option.data_dir = option.root_dir + f"/{option.dataset_name}"
                    option.bias_split_name = 'full_v1'
                    option.trainval_sub_dir = option.bias_split_name + "_" + str(p_bias)
                    option.target_name = 'digit'
                    option.bias_variables = bias_variables
                    option.bias_variable_name = 'group_ix'  # For RUBi
                    option.lr = lr
                    option.weight_decay = wd

                    option.bias_model_hid_dims = 64  # RUBi
                    option.bias_predictor_hid_dims = 64  # LNL
                    option.num_classes = 10

                    # Assumption: num_groups, bias_variable_dims and num_bias_classes will be set dynamically when creating the dataset

                    # Optimizer + Model
                    set_if_null(option, 'optimizer_name', 'Adam')
                    set_if_null(option, 'batch_size', 128)
                    set_if_null(option, 'epochs', 50)
                    set_if_null(option, 'model_name', 'ResNet10')

                    option.bias_variables_str = '+'.join(option.bias_variables)
                    # Configure name of the experiments and directory to save the results
                    if option.save_dir is None:
                        option.save_dir = os.path.join(option.root_dir, option.project_name, option.trainval_sub_dir,
                                                       f'target_{option.target_name}_bias_{option.bias_variables_str}',
                                                       option.trainer_name)
                    if option.expt_name is None:
                        option.expt_name = f'lr_{lr}_wd_{wd}'

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

                    run(option)


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
