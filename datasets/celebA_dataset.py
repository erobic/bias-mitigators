import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
from utils.data_utils import dict_collate_fn
import logging
import shutil
from option import ROOT


class CelebADataset(Dataset):
    """
    CelebA dataset (already cropped and centered). This code is adapted from: https://github.com/kohpangwei/group_DRO.
    """

    def __init__(self, data_dir, target_name, bias_variable_names, augment_data=False, no_image=False):
        self.data_dir = data_dir
        self.target_name = target_name
        self.bias_variable_names = bias_variable_names
        self.augment_data = augment_data
        self.no_image = no_image

        # Read in attributes.
        col_names = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                     'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                     'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                     'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                     'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                     'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        self.attributes = pd.read_csv(
            os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt'), sep='\s+', names=col_names, skiprows=2)

        # Split out image ids and attribute names
        self.data_dir = os.path.join(self.data_dir, 'Img', 'img_align_celeba')
        self.image_ids = self.attributes['image_id'].values
        self.attributes = self.attributes.drop(labels='image_id', axis='columns')
        self.attribute_names = self.attributes.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attribute_vals = self.attributes.values
        self.attribute_vals[self.attribute_vals == -1] = 0

        # Get the y values
        target_location = self.get_attribute_location(self.target_name)
        self.y_array = self.attribute_vals[:, target_location]
        self.num_classes = 2

        # Map the bias variables to a number 0,...,2^|confounder_idx|-1
        self.bias_variable_idx = [self.get_attribute_location(a) for a in self.bias_variable_names]
        self.num_bias_variables = len(self.bias_variable_idx)
        bias_variables = self.attribute_vals[:, self.bias_variable_idx]
        bias_variable_id = bias_variables @ np.power(2, np.arange(len(self.bias_variable_idx)))
        self.bias_variable_ixs = bias_variable_id

        # Map to groups
        # Note, we are grouping things by label and bias variable
        self.num_groups = self.num_classes * pow(2, len(self.bias_variable_idx))
        self.group_array = (self.y_array * (self.num_groups / 2) + self.bias_variable_ixs).astype('int')

        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(data_dir, 'Eval', 'list_eval_partition.txt'), sep='\s+', names=['image_id', 'partition'])
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.features_mat = None
        self.train_transform = get_transform_celebA(train=True, augment_data=augment_data)
        self.eval_transform = get_transform_celebA(train=False, augment_data=augment_data)

    def __len__(self):
        return len(self.image_ids)

    def get_attribute_location(self, attr_name):
        return self.attribute_names.get_loc(attr_name)

    def assign_group_info(self, ix, ret_obj):
        # useful_attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        #                      'Bangs',
        #                      'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
        #                      'Bushy_Eyebrows',
        #                      'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
        #                      'High_Cheekbones',
        #                      'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
        #                      'Pale_Skin',
        #                      'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
        #                      'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        #                      'Wearing_Necktie', 'Young']
        useful_attributes = ['Blond_Hair', 'Male']
        for useful_attr in useful_attributes:
            attr_location = self.get_attribute_location(useful_attr)
            attr_val = self.attribute_vals[ix, attr_location]
            ret_obj[useful_attr + "_group_name"] = useful_attr + f":{attr_val}_y:{self.y_array[ix]}_group_name"

    def __getitem__(self, ix):
        ix = int(ix)
        y = self.y_array[ix]
        group_ix = self.group_array[ix]
        img_filename = os.path.join(self.data_dir, self.image_ids[ix])

        if not self.no_image:
            img = Image.open(img_filename).convert('RGB')

            # Figure out split and transform accordingly
            if self.split_array[ix] == self.split_dict['train'] and self.train_transform:
                img = self.train_transform(img)
            elif (self.split_array[ix] in [self.split_dict['val'], self.split_dict['test']] and
                  self.eval_transform):
                img = self.eval_transform(img)
            x = img
        else:
            x = None

        #
        ret_obj = {'x': x,
                   'y': y,
                   'group_ix': group_ix,
                   'dataset_ix': ix,
                   'filename': img_filename,
                   }
        self.assign_group_info(ix, ret_obj)

        # Add bias variables
        for bias_name in self.bias_variable_names:
            bias_val = self.attributes[bias_name].values[ix]
            ret_obj[bias_name] = bias_val
        ret_obj[self.target_name] = self.attributes[self.target_name].values[ix]

        # Add group_name
        ret_obj['group_name'] = self.group_str(group_ix)
        return ret_obj

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test'), split + ' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.num_groups / self.num_classes)
        c = group_idx % (self.num_groups // self.num_classes)

        if self.target_name == 'Blond_Hair' and len(self.bias_variable_names) == 1 and self.bias_variable_names[
            0] == 'Male':
            target_names = ['Non-Blond', 'Blond']
            attr_names = ['Non-Male', 'Male']
            group_name = target_names[int(y)] + ' ' + attr_names[int(c)]
        else:
            group_name = f'{self.target_name} = {int(y)}'
            bin_str = format(int(c), f'0{self.num_bias_variables}b')[::-1]
            for attr_idx, attr_name in enumerate(self.bias_variable_names):
                group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


def get_transform_celebA(train, augment_data, target_resolution=(224, 224)):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def create_celebA_dataset(data_dir, split, target_name, bias_variable_names, filter=None,
                          limit=None, ratio=None, no_image=False):
    col_names = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                 'Bangs',
                 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                 'Wearing_Necktie', 'Young']
    attrs_df = pd.read_csv(
        os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt'), sep='\s+', names=col_names, skiprows=2)
    split_df = pd.read_csv(
        os.path.join(data_dir, 'Eval', 'list_eval_partition.txt'), sep='\s+', names=['image_id', 'partition'])
    split_array = split_df['partition'].values
    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }

    # Gather indices for current data split
    split_mask = split_array == split_dict[split]
    split_indices = np.where(split_mask)[0]

    # Filter data based on specified filter/limit
    filtered_indices_list = []
    if filter is not None:
        for key in filter:
            key_values = attrs_df[key].values
            filter_mask = key_values == filter[key]
            filtered_indices = np.where(filter_mask)[0]
            filtered_indices_list.append(filtered_indices)

    final_filtered_indices = split_indices
    for curr_filtered_indices in filtered_indices_list:
        final_filtered_indices = np.intersect1d(final_filtered_indices, curr_filtered_indices)
    if limit is not None:
        final_filtered_indices = final_filtered_indices[0:min(len(final_filtered_indices), limit)]
    if ratio is not None:
        np.random.shuffle(final_filtered_indices)
        filter_len = int(len(final_filtered_indices) * ratio)
        final_filtered_indices = final_filtered_indices[0:filter_len]
        logging.getLogger().info(f"Length of dataset {len(final_filtered_indices)}")
    dataset = CelebADataset(data_dir, target_name=target_name, bias_variable_names=bias_variable_names,
                            augment_data=False,
                            no_image=no_image)
    dataset = Subset(dataset, final_filtered_indices)
    return dataset


def create_celebA_datasets(data_dir, target_name, bias_variables, filter=None, limit=None, train_ratio=None):
    train_set = create_celebA_dataset(data_dir, 'train', target_name, bias_variables, filter, limit,
                                      ratio=train_ratio)
    val_set = create_celebA_dataset(data_dir, 'val', target_name, bias_variables, filter, limit)
    test_set = create_celebA_dataset(data_dir, 'test', target_name, bias_variables, filter, limit)
    return train_set, val_set, test_set


def create_celebA_dataloaders(option):
    """
    Uses the train loader for training and train, val and test splits for testing
    :param option:
    :return:
    """
    train_set, val_set, test_set = create_celebA_datasets(option.data_dir, option.target_name,
                                                          bias_variables=option.bias_variables,
                                                          train_ratio=option.train_ratio)
    train_loader = DataLoader(train_set, batch_size=option.batch_size, shuffle=True, num_workers=option.num_workers,
                              collate_fn=dict_collate_fn())
    val_loader = DataLoader(val_set, batch_size=option.batch_size, shuffle=False, num_workers=option.num_workers,
                            collate_fn=dict_collate_fn())
    test_loader = DataLoader(test_set, batch_size=option.batch_size, shuffle=False, num_workers=option.num_workers,
                             collate_fn=dict_collate_fn())
    return {
        'Train': train_loader,
        'Test': {
            'Train': train_loader,
            'Val': val_loader,
            'Test': test_loader
        }
    }


def compute_frequency(loader):
    group_name_to_freq = {}
    for batch in loader:
        gns = batch['group_name']
        for gn in gns:
            if gn not in group_name_to_freq:
                group_name_to_freq[gn] = 0
            group_name_to_freq[gn] += 1
    print(group_name_to_freq)

# if __name__ == "__main__":
#     for bias in ['Male', 'Chubby', 'Eyeglasses', 'Heavy_Makeup']:
#         dataset = create_celebA_dataset(f'{ROOT}/CelebA', 'test', target_name='Attractive',
#                                         bias_variable_names=[bias], no_image=True)
#         loader = torch.utils.data.DataLoader(dataset, 128, collate_fn=dict_collate_fn(), num_workers=0)
#         compute_frequency(loader)
