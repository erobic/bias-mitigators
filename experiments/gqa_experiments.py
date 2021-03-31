import os


def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)


def gqa_experiments(option, run):
    # Method-specific arguments are defined in the bash files inside: scripts/gqa-ood

    # Here, we configure rest of the arguments which likely DO NOT NEED TO BE CHANGED
    option.dataset_name = 'GQA'
    option.data_dir = option.root_dir + f"/{option.dataset_name}"
    option.train_ratio = None  # set to sth like 0.1 for debugging

    # Define the bias and target variables
    # option.num_bias_classes: # This is set to num_groups in main.py for GQA
    # option.num_classes: # This is set by gqa_dataset
    # option.num_groups: # This is set by gqa_dataset

    # Optimizer + Model
    set_if_null(option, 'model_name', 'UpDn')
    set_if_null(option, 'optimizer_name', 'Adam')
    set_if_null(option, 'batch_size', 128)
    set_if_null(option, 'epochs', 30)

    # We allow training on subset of data using train_ratio argument. If it is "None", then the full set is used.
    if option.train_ratio is not None:
        dataset_name = f"{option.dataset_name}_ratio_{option.train_ratio}"
    else:
        dataset_name = option.dataset_name

    # Configure name of the experiments and directory to save the results
    if option.save_dir is None:
        option.save_dir = os.path.join(option.root_dir, option.project_name, dataset_name, option.trainer_name)

    if option.expt_name is None:
        option.expt_name = f"lr_{option.lr}_wd_{option.weight_decay}"

    if option.key_to_group_by is not None:
        option.expt_name += f'_expl_bias_{option.key_to_group_by}'

    # Method-specific configurations
    if option.trainer_name == 'GroupDROTrainer':
        option.balanced_sampling_attributes = ['group_ix']  # Perform balanced sampling
        option.group_by = 'group_ix'  # Groups for GDRO
        option.key_to_group_by = 'group_name'  # Used to find readable group names

    if option.trainer_name == 'RUBiTrainer':
        set_if_null(option, 'bias_model_hid_dims', 2048)
        # We experimented with MLP2 and MLP3 too, but MLP1 had worked best in the preliminary experiments
        set_if_null(option, 'bias_model_name', 'MLP3')
        set_if_null(option, 'bias_variable_type', 'categorical')

    if option.trainer_name == 'LNLTrainer':
        # We experimented with MLP2 and MLP3 too, but MLP1 had worked best in the preliminary experiments
        set_if_null(option, 'bias_predictor_name', 'MLP3')
        set_if_null(option, 'bias_predictor_in_layer', 'question_features')
        option.bias_predictor_in_dims = 1024
        option.bias_predictor_hid_dims = 1024

    if option.trainer_name == 'IRMv1Trainer':
        set_if_null(option, 'num_envs_per_batch', 16)
    if option.bias_variable_name != 'question_features':
        option.bias_variable_type = 'categorical'

    # Test epochs
    option.test_every = 15
    option.save_every = 30
    option.save_model_every = 30

    if option.model_name == 'MCAN':
        option.bias_variable_dims = 2048
    else:
        option.bias_variable_dims = 1024

    run(option)
