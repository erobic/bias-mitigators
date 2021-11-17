import logging
from multiprocessing import Process

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import os
import json
# from datasets.mnist_utils import *
# from datasets.texture_utils import *
import numpy as np
from utils.data_utils import dict_collate_fn
from datasets.biased_mnist_generator import BiasedMNISTGenerator


# from datasets.class_imbalance_utils import *


class BiasedMNISTDataset(Dataset):
    def __init__(self, data_dir, bias_split_name, trainval_sub_dir, split, target_name, bias_variable_names):
        """

        :param data_dir: Directory where the images and factor info are located
        :param bias_split_name: Name of specific subset within biased mnist e.g., digit_vs_digit_color
        :param trainval_sub_dir: Name of the sub-split. Each bias_split (e.g., long_tailed_mnist) may contain multiple sub-splits e.g., long_tailed_mnist_0.01
        :param split: train or val or test
        :param target_name: What do want to predict? e.g., digit, texture_ix, color_ix
        :param bias_variable_names: Samples will be grouped using these variables.
        """
        super(BiasedMNISTDataset, self).__init__()
        self.data_dir = data_dir
        self.bias_split_name = bias_split_name
        self.trainval_sub_dir = trainval_sub_dir
        self.split = split
        self.target_name = target_name
        self.bias_variable_names = bias_variable_names
        self.num_classes = 10

        # Load the data files
        self.load_factor_config()
        self.prepare_dataset()
        self.num_groups = self.main_group_utils.max_group_ix

    def load_factor_config(self):
        if 'train' in self.split or 'val' in self.split:
            split_name = 'trainval'
            sub_dir = self.trainval_sub_dir
        else:
            split_name = 'test'
            sub_dir = f'{self.bias_split_name}'

        self.images_dir = os.path.join(self.data_dir, sub_dir, split_name)
        self.factors_data = json.load(open(os.path.join(self.data_dir, sub_dir, f'{split_name}.json')))
        # if split_name == 'test':
        #     count_groups(self.factors_data)

        self.main_group_utils = GroupUtils(self.target_name, self.bias_variable_names)

    def __len__(self):
        return len(self.factors_data)

    def prepare_dataset(self):
        """
        Assigns each data item to a group
        :return:
        """
        self.data_items = {}
        bias_variable_to_group_utils = {}

        for index, curr_factor_to_val in enumerate(self.factors_data):
            curr_factor_to_val = self.factors_data[index]  # Contains exact values for each factor
            group_ix, group_name, maj_min_group_ix, maj_min_group_name = self.main_group_utils.to_group_ix_and_name(
                curr_factor_to_val)
            # Gather the class id of the target attribute and add group based on bias variable
            y = curr_factor_to_val[self.target_name]
            item_data = {
                'y': y,
                'dataset_ix': curr_factor_to_val['index'],
                'group_ix': group_ix,
                'group_name': group_name,
                'maj_min_group_ix': maj_min_group_ix,
                'maj_min_group_name': maj_min_group_name
            }

            for bias_variable in BiasedMNISTGenerator.FACTORS_v1:
                if bias_variable not in bias_variable_to_group_utils:
                    bias_variable_to_group_utils[bias_variable] = GroupUtils(target_name=self.target_name,
                                                                             bias_variable_names=[bias_variable],
                                                                             num_classes=self.num_classes)
                _, _, bv_group_ix, bv_group_name = bias_variable_to_group_utils[bias_variable].to_group_ix_and_name(
                    curr_factor_to_val)
                item_data[f'{bias_variable}_group_ix'] = bv_group_ix
                item_data[
                    f'{bias_variable}_group_name'] = "zzz_" + bv_group_name  # prefixed with z to make it easily viewable

            self.data_items[curr_factor_to_val['index']] = item_data

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.images_dir, f'{index}.jpg'))
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((160, 160, 3))
            index = 0
        img = torch.FloatTensor(img)
        img = img.permute((2, 0, 1))
        item_data = self.data_items[index]
        item_data['x'] = img
        return item_data


def create_biased_mnist_datasets(data_dir, bias_split_name, trainval_sub_dir, target_name, bias_variable_names):
    # First see if train_ixs are defined within the sub_dir, else use the default one
    if os.path.exists(os.path.join(data_dir, trainval_sub_dir, 'train_ixs.json')):
        train_ixs = json.load(open(os.path.join(data_dir, trainval_sub_dir, 'train_ixs.json')))
        val_ixs = json.load(open(os.path.join(data_dir, trainval_sub_dir, 'val_ixs.json')))
    else:
        train_ixs = json.load(open(os.path.join(data_dir, 'train_ixs.json')))
        val_ixs = json.load(open(os.path.join(data_dir, 'val_ixs.json')))

    train_set = BiasedMNISTDataset(data_dir, bias_split_name, trainval_sub_dir, 'train', target_name=target_name,
                                   bias_variable_names=bias_variable_names)
    num_groups = train_set.num_groups
    train_set = Subset(train_set, train_ixs)

    val_set = BiasedMNISTDataset(data_dir, bias_split_name, trainval_sub_dir, 'val', target_name=target_name,
                                 bias_variable_names=bias_variable_names)
    # balanced_val_set = BiasedMNISTDataset(data_dir, bias_split_name, f'{bias_split_name}_0.1', 'val',
    #                                       target_name=target_name,
    #                                       bias_variable_names=bias_variable_names)
    val_set = Subset(val_set, val_ixs)
    # balanced_val_set = Subset(balanced_val_set, val_ixs)

    # balanced_val_set = BiasedMNISTDataset(data_dir, bias_split_name, 0.1, 'val',
    #                                       target_name=target_name, bias_variable_names=bias_variable_names)
    # balanced_val_set = Subset(balanced_val_set, val_ixs)

    test_set = BiasedMNISTDataset(data_dir, bias_split_name, trainval_sub_dir, 'test', target_name=target_name,
                                  bias_variable_names=bias_variable_names)
    return train_set, val_set, test_set, num_groups


def create_biased_mnist_dataloaders(config):
    train_set, val_set, test_set, num_groups = create_biased_mnist_datasets(config.data_dir,
                                                                            config.bias_split_name,
                                                                            config.trainval_sub_dir,
                                                                            config.target_name,
                                                                            config.bias_variables)
    logging.getLogger().info(f"Setting the num_groups to {num_groups}")
    config.num_groups = num_groups
    config.bias_variable_dims = num_groups
    config.num_bias_classes = num_groups

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=dict_collate_fn())
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=dict_collate_fn())
    # balanced_val_loader = DataLoader(balanced_val_set, batch_size=config.dataset.batch_size, shuffle=False,
    #                                  num_workers=config.dataset.num_workers, collate_fn=dict_collate_fn())
    # balanced_val_loader = DataLoader(balanced_val_set, batch_size=config.dataset.batch_size, shuffle=False,
    #                                  num_workers=config.dataset.num_workers, collate_fn=dict_collate_fn())
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dict_collate_fn())
    return {
        'Train': train_loader,
        'Test': {
            'Train': train_loader,
            'Val': val_loader,
            # 'Balanced Val': balanced_val_loader,
            'Test': test_loader
        }
    }


class GroupUtils():
    def __init__(self, target_name, bias_variable_names, num_classes=10, use_majority_minority_grouping=False):
        """
        Groups the data based on the specified bias variables i.e., each unique combination of bias variables is a group
        e.g., group#1 = (digit color=red, texture=+, texture color=green etc.)

        If use_majority_minority_grouping is used

        :param target_name: Variable to predict
        :param bias_variable_names: List of variables that act as biases
        :param num_classes:
        :param target_name:
        :param use_majority_minority_grouping:
        """
        self.num_classes = num_classes
        self.target_name = target_name
        self.bias_variable_names = bias_variable_names
        self.use_majority_minority_grouping = use_majority_minority_grouping
        self.group_name_to_ix = {}
        self.maj_min_group_name_to_ix = {}
        self.max_group_ix = 0
        self.max_maj_min_group_ix = 0

    def to_group_ix_and_name(self, curr_factor_to_val, should_print=False):
        group_name_parts = []

        # Assume that if the factor ix is same as the index of the factor val, then it is a majority group,
        # else it is a minority group
        maj_min_group_name_parts = []
        class_ix = curr_factor_to_val[self.target_name]

        # Go through all of the bias variables, to come up with the group name
        for ix, bias_name in enumerate(self.bias_variable_names):
            if bias_name == 'digit':
                bias_ix_val = curr_factor_to_val[bias_name]
            else:
                bias_ix_val = curr_factor_to_val[bias_name + "_ix"]
            maj_min = 'minority'
            if bias_ix_val == class_ix:
                maj_min = 'majority'
            group_name_parts.append(f'{bias_name}_{bias_ix_val}')
            maj_min_group_name_parts.append(f'{bias_name}_{maj_min}')

        group_name = '+'.join(group_name_parts)
        maj_min_group_name = '+'.join(maj_min_group_name_parts)
        if self.use_majority_minority_grouping:
            group_name = maj_min_group_name

        if group_name not in self.group_name_to_ix:
            self.group_name_to_ix[group_name] = self.max_group_ix
            self.max_group_ix += 1
            if should_print:
                print(f"adding group name {group_name}, max_group_ix: {self.max_group_ix}")

        if maj_min_group_name not in self.maj_min_group_name_to_ix:
            self.maj_min_group_name_to_ix[maj_min_group_name] = self.max_maj_min_group_ix
            self.max_maj_min_group_ix += 1
        group_ix = self.group_name_to_ix[group_name]
        maj_min_group_ix = self.maj_min_group_name_to_ix[maj_min_group_name]
        return group_ix, group_name, maj_min_group_ix, maj_min_group_name


if __name__ == "__main__":
    datasets = create_biased_mnist_datasets('/hdd/robik/biased_mnist', bias_split_name='full',
                                            trainval_sub_dir='full_0.9', target_name='digit',
                                            bias_variable_names=['texture_ix', 'texture_color_ix', 'digit_color_ix',
                                                                 'digit_scale_ix', 'digit'])
    trainset = datasets[0]
    testset = datasets[2]
    # dataloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0, collate_fn=dict_collate_fn())
    # class_ix_to_cnt = {}
    # for batch in dataloader:
    #     for y in batch['y']:
    #         if y not in class_ix_to_cnt:
    #             class_ix_to_cnt[y] = 0
    #         class_ix_to_cnt[y] += 1
    # print(class_ix_to_cnt)
