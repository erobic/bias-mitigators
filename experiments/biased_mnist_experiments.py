import os


def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)


def celebA_experiments(option, run):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/celebA

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    option.dataset_name = 'CelebA'
    option.data_dir = option.root_dir + f"/{option.dataset_name}"

    option.train_ratio = None # None, set to sth like 0.1 for debugging

    # Define the bias and target variables
    option.target_name = 'Blond_Hair'
    option.bias_variables = ['Male']
    option.bias_variable_name = 'Male'
    option.num_bias_classes = 2
    option.num_classes = 2
    option.num_groups = 4

    # Optimizer + Model
    set_if_null(option, 'optimizer_name', 'SGD')
    set_if_null(option, 'batch_size', 128)
    set_if_null(option, 'epochs', 50)
    set_if_null(option, 'model_name', 'ResNet18')

    # We allow training on subset of data using train_ratio argument. If it is "None", then the full set is used.
    if option.train_ratio is not None:
        dataset_name = f"{option.dataset_name}_ratio_{option.train_ratio}"
    else:
        dataset_name = option.dataset_name

    # Configure name of the experiments and directory to save the results
    if option.save_dir is None:
        option.save_dir = os.path.join(option.root_dir, option.project_name, dataset_name, 'predict_' + option.target_name,
                                       option.trainer_name)
    if option.expt_name is None:
        option.expt_name = f"lr_{option.lr}_wd_{option.weight_decay}"

    # Method-specific configurations

    if option.trainer_name == 'GroupDROTrainer':
        option.balanced_sampling_attributes = ['group_ix']  # Perform balanced sampling
        option.group_by = 'group_ix'  # Groups for GDRO
        option.key_to_group_by = 'group_name'  # Used to find readable group names

    if option.trainer_name == 'RUBiTrainer':
        set_if_null(option, 'bias_model_hid_dims', 512)
        # We experimented with MLP2 and MLP3 too, but MLP1 had worked best in the preliminary experiments
        set_if_null(option, 'bias_model_name', 'MLP1')
        set_if_null(option, 'bias_variable_type', 'categorical')
        option.bias_variable_dims = 2

    if option.trainer_name == 'LNLTrainer':
        # We experimented with MLP2 and MLP3 too, but MLP1 had worked best in the preliminary experiments
        feature_dims = get_feature_dims(option.model_name, option.num_classes)
        set_if_null(option, 'bias_predictor_name', 'MLP1')
        set_if_null(option, 'bias_predictor_in_layer', 'model.pooled2')
        option.bias_predictor_in_dims = feature_dims[option.bias_predictor_in_layer]
        option.bias_predictor_hid_dims = feature_dims[option.bias_predictor_in_layer]

    if option.trainer_name == 'IRMv1Trainer':
        set_if_null(option, 'num_envs_per_batch', 4)

    if option.trainer_name == 'SpectralDecouplingTrainer':
        option.spectral_decoupling_lambdas = [10.0, 10.0]  # For CelebA, we have per-class lambdas and gammas for SD.
        option.spectral_decoupling_gammas = [0.44, 2.5]

    # Test epochs
    option.test_epochs = [e for e in range(40, 51)]  # Due to instability, we average accuracies over the last 10 epochs
    option.test_every = 10  # We further test every 10 epochs
    option.save_every = 50

    run(option)


def get_feature_dims(model_name, num_classes):
    if 'ResNet18' in model_name:
        return {
            'model.conv1': 64,
            'model.pooled_conv1': 64,
            'model.layer1.1.conv2': 64,
            'model.pooled1': 64,
            'model.layer2.1.conv2': 128,
            'model.layer2_flattened': 128 * 28 * 28,
            'model.pooled2': 128,
            'model.layer3.1.conv2': 256,
            'model.pooled3': 256,
            'model.layer4.1.conv2': 512,
            'model.pooled4': 512,
            'model.fc': 10,
            'logits': num_classes
        }
