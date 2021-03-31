from datasets.celebA_dataset import create_celebA_dataloaders
import logging
import torch
from torch.utils.data import DataLoader
import json
from datasets.vqa.gqa_dataset import GQADataset, create_gqa_dataloaders


def build_balanced_loader(dataloader, balanced_sampling_attributes=['y'], balanced_sampling_gamma=1, replacement=True):
    logger = logging.getLogger()
    all_group_names = []

    # Count frequencies for all groups of attributes to balance,
    # and assign each sample to a group, so that we can compute its sampling weight later on
    group_name_to_count = {}
    for batch in dataloader:
        batch_group_names = []
        for ix, _ in enumerate(batch['y']):
            group_name = ""
            for attr in balanced_sampling_attributes:
                group_name += f"{attr}_{batch[attr][ix]}_"
            batch_group_names.append(group_name)

        for group_name in batch_group_names:
            if group_name not in group_name_to_count:
                group_name_to_count[group_name] = 0
            group_name_to_count[group_name] += 1
            all_group_names.append(group_name)

    # Create the balanced loader
    weights = []
    for val in all_group_names:
        weights.append(1 / group_name_to_count[val] ** balanced_sampling_gamma)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights),
                                                              replacement=replacement)
    balanced_dataloader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=weighted_sampler,
                                     num_workers=dataloader.num_workers, collate_fn=dataloader.collate_fn)
    logger.info(f"Created balanced loader for {len(weights)} samples of dataset size {len(dataloader.dataset)}")
    logger.info(f"Group counts: {json.dumps(group_name_to_count, indent=4)}")
    return balanced_dataloader


def build_dataloaders(option):
    dataset_name = option.dataset_name.lower()
    if dataset_name == 'biased_mnist':
        loaders = create_biased_mnist_dataloaders(option)  # Sets the num_groups
    elif dataset_name == 'celeba':
        loaders = create_celebA_dataloaders(option)
    elif dataset_name == 'gqa':
        loaders = create_gqa_dataloaders(option)
    loaders['Unbalanced Train'] = loaders['Train']
    if option.balanced_sampling_attributes is not None:
        loaders['Train'] = build_balanced_loader(loaders['Train'],
                                                 option.balanced_sampling_attributes,
                                                 balanced_sampling_gamma=option.balanced_sampling_gamma,
                                                 replacement=True)
    return loaders