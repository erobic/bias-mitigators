import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import cv2
from utils.data_utils import dict_collate_fn
from PIL import Image
import copy
from torchvision.datasets.mnist import MNIST
import json
import option
from datasets.shape_generator import ShapeGenerator
import itertools


def dataset_to_xy(dataset, width=32, height=32):
    images = []
    labels = []
    for img, label in dataset:
        img = img.numpy().transpose(1, 2, 0)
        img = np.repeat(img, 3, axis=2)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) * 255
        images.append(img)
        labels.append(label)
    return images, labels


class FactorToValues():
    def __init__(self):
        # For each factor, we assume that the value corresponding to its class index is the biased factor
        # and the rest are the non-biased factors.
        digit_colors = np.asarray([(255, 0, 0),
                                   (0, 255, 0),
                                   (0, 0, 255),
                                   (255, 255, 0),
                                   (0, 255, 255),
                                   (255, 0, 255),
                                   (128, 255, 0),
                                   (128, 0, 255),
                                   (0, 128, 255),
                                   (0, 255, 128)])
        texture_colors = (255 - digit_colors)
        bg_colors = (texture_colors / 4).astype(np.uint8).tolist()
        texture_colors = texture_colors.tolist()
        distractor_fg_colors = np.flip(digit_colors, 0).tolist()
        digit_colors = digit_colors.tolist()
        distractors = ['circle', 'right_triangle', 'obtuse_triangle', 'square', 'parallelogram', 'kite',
                       'pentagon', 'semi_circle', 'plus', 'arrow']
        digit_cell_number = np.arange(0, 9, 1).tolist()
        digit_cell_number.append(4)
        # assuming that the digit image size is 1x1, it specifies the amount of shift distractor should have
        textures = np.asarray(['-', '/', '|', '+', 'x', '~', '::', '>', '<', '\\'])
        textures = textures.tolist()

        self.factor_to_values = {
            'digit_color': digit_colors,
            'distractor_color': distractor_fg_colors,
            'bg_color': bg_colors,
            'distractor': distractors,
            'texture': textures,
            'texture_color': texture_colors,
            'digit_cell_number': digit_cell_number
        }
        self.ordered_factors = ['bg_color', 'digit_color', 'digit_cell_number', 'distractor', 'distractor_color',
                                'texture',
                                'texture_color']
        self.factor_to_labels = {
            'digit_color': 'Digit Color',
            'distractor_color': 'Distractor Color',
            'bg_color': 'Background Color',
            'distractor': 'Distractor Shape',
            'texture': 'Texture Pattern',
            'texture_color': 'Texture Color',
            'digit_cell_number': 'Digit Position'
        }

        factor_value_to_ix = {}
        for factor in self.factor_to_values:
            if factor not in factor_value_to_ix:
                factor_value_to_ix[factor] = {}
            vals = self.factor_to_values[factor]
            val_ix = 0
            for val in vals:
                val = str(val)
                factor_value_to_ix[factor][val] = val_ix
                val_ix += 1
        self.factor_value_to_ix = factor_value_to_ix


class BiasedMNISTGenerator():
    """
    Each image is a 3x3 grid, with one of the MNIST digits present at one of the grid cells and geometric shapes present at other locations.
    The MNIST digit is correlated with multiple factors: fg color, bg color, shapes, fg texture, bg texture and location
    """

    def __init__(self,
                 original_mnist_dir,
                 split,
                 bias_probabilities):
        self.split = split
        self.original_mnist_dir = original_mnist_dir
        self.bias_probabilities = bias_probabilities
        self.num_classes = 10
        if split == 'train':
            self.original_dataset = datasets.MNIST(self.original_mnist_dir, train=True, download=True,
                                                   transform=transforms.ToTensor())
            self.filename = 'trainval'
        else:
            self.original_dataset = datasets.MNIST(self.original_mnist_dir, train=False, download=True,
                                                   transform=transforms.ToTensor())
            self.filename = self.split
        self.mnist_dim = 32
        self.grid_dim = 3
        self.biased_mnist_dim = self.mnist_dim * self.grid_dim

    def __init(self):
        self.images, self.labels = dataset_to_xy(self.original_dataset)

        self.__init_grid_id_to_xy()

        self.__init_factor_to_values()
        self.shape_generator = ShapeGenerator(dim=self.mnist_dim)

    def __init_grid_id_to_xy(self):
        self.grid_id_to_xy = {}
        self.xy_to_grid_id = {}
        grid_id = 0
        for x in np.arange(0, self.grid_dim):
            for y in np.arange(0, self.grid_dim):
                # Gives the location of top, left corner of the cell
                self.grid_id_to_xy[grid_id] = {
                    'x': x * self.mnist_dim,
                    'y': y * self.mnist_dim
                }
                self.xy_to_grid_id[f'x_{x}_y_{y}'] = grid_id
                grid_id += 1

    def __init_factor_to_values(self):
        obj = FactorToValues()
        self.factor_to_values = obj.factor_to_values

    def get_factor_to_non_biased_values(self):
        """
        For each class within each factor, this function returns the non-biased values for that factor
        :return:
        """
        factor_to_non_biased_values = {}
        for bias_factor in self.factor_to_values:
            factor_to_non_biased_values[bias_factor] = {}
            for class_ix in np.arange(0, self.num_classes):
                non_biased_vals = copy.deepcopy(self.factor_to_values[bias_factor])
                del non_biased_vals[class_ix]  # Remove the biased value
                factor_to_non_biased_values[bias_factor][class_ix] = non_biased_vals
        return factor_to_non_biased_values

    def sample_factors(self):
        """
        For each (x, y) data point, it samples each of the bias factors to apply, based on the specified bias probabilities

        :param original_dataset: MNIST dataset containing the images and labels
        :param save_dir: The sampled factors are saved to this directory
        :return:
        """
        logging.getLogger().info("Sampling values for each factor of variation")

        # For a given sample, if bias_rand <= p_bias, then the correlated/biased value for that factor is used
        # Else, we randomly sample from the remaining values
        bias_rand, non_bias_rands = {}, {}
        factor_to_non_biased_values = self.get_factor_to_non_biased_values()

        # Compute random numbers to determine the value for each factor
        for bias_factor in self.factor_to_values:
            # Sample whether to keep the biased factor or sample uniformly from the rest
            bias_rand[bias_factor] = np.random.random(len(self.labels))
            non_bias_rands[bias_factor] = np.random.random((len(self.labels), self.num_classes - 1))

        # Choose the values for each factor
        sampled_factors = []
        for ix, lbl in enumerate(self.labels):
            sampled_factor = {
                'index': ix,
                'label': lbl}

            for bias_factor in self.factor_to_values:
                if bias_factor == 'distractor':
                    if bias_rand[bias_factor][ix] <= self.bias_probabilities[bias_factor]:
                        primary_distractor_ix = lbl
                        primary_distractor = self.factor_to_values[bias_factor][primary_distractor_ix]
                    else:
                        primary_distractor = np.random.choice(factor_to_non_biased_values[bias_factor][lbl])
                        primary_distractor_ix = self.factor_to_values[bias_factor].index(primary_distractor)

                    # Assign the primary distractor shape to each cell
                    sampled_factor['primary_distractor'] = primary_distractor
                    sampled_factor[bias_factor] = [primary_distractor] * (self.grid_dim ** 2)

                    # Assign a distractor shape to each cell, giving high probability to the primary distractor shape
                    # primary_distractor_proba = self.bias_probabilities[bias_factor]
                    # num_values = len(self.factor_to_values[bias_factor])
                    # secondary_distractor_proba = (1 - primary_distractor_proba) / (num_values - 1)
                    # distractor_probas = np.ones(num_values) * secondary_distractor_proba
                    # distractor_probas[primary_distractor_ix] = primary_distractor_proba
                    # sampled_factor[bias_factor] = np.random.choice(self.factor_to_values[bias_factor],
                    #                                                self.grid_dim * self.grid_dim,
                    #                                                p=distractor_probas).tolist()
                else:

                    if bias_rand[bias_factor][ix] <= self.bias_probabilities[bias_factor]:
                        # Assign the biased value for each factor
                        sampled_factor[bias_factor] = self.factor_to_values[bias_factor][lbl]
                    else:
                        # Sample uniformly from rest of the factors
                        rand_ix = int(np.random.choice(len(factor_to_non_biased_values[bias_factor][lbl]), size=1))
                        sampled_factor[bias_factor] = factor_to_non_biased_values[bias_factor][lbl][rand_ix]

            sampled_factors.append(sampled_factor)
        return sampled_factors

    def create_and_load_sampled_factors(self, save_dir, rewrite=False):
        factor_json_file = os.path.join(save_dir, f'{self.filename}.json')
        if not os.path.exists(factor_json_file) or rewrite:
            print("Sampling the factors for this dataset...")
            sampled_factors = self.sample_factors()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, f'{self.filename}.json'), 'w') as f:
                json.dump(sampled_factors, f, indent=4, sort_keys=True)

        sampled_factors = json.load(open(factor_json_file))
        return sampled_factors

    def __create_new_image(self, bg_color):
        biased_mnist_img = np.ones((self.biased_mnist_dim, self.biased_mnist_dim, 3))
        for channel in [0, 1, 2]:
            biased_mnist_img[:, :, channel] *= bg_color[channel]
        return biased_mnist_img

    def __get_fg_mask(self, digit_img):
        mask = np.where(digit_img > 127, np.ones_like(digit_img), np.zeros_like(digit_img))
        return mask

    def __draw_shape(self, full_img, x1, y1, x2, y2, mask, color):
        for channel in [0, 1, 2]:
            full_img[x1:x2, y1:y2, channel] = np.where(mask[:, :, channel] > 0,
                                                       mask[:, :, channel] * color[channel],
                                                       full_img[x1:x2, y1:y2, channel])
        return full_img

    def __draw_distractor_shapes(self, full_img, distractors, color, digit_cell_number):
        for cell_num, distractor in enumerate(distractors):
            if cell_num == digit_cell_number:
                continue
            xy = self.grid_id_to_xy[cell_num]
            x1, y1 = xy['x'], xy['y']
            x2, y2 = x1 + self.mnist_dim, y1 + self.mnist_dim
            shape = self.shape_generator.generate(distractor) / 255
            self.__draw_shape(full_img, x1, y1, x2, y2, shape, color)
        return full_img

    def __draw_digit(self, full_img, digit_mask, digit_color, digit_position):
        x1, y1 = digit_position['x'], digit_position['y']
        x2, y2 = x1 + self.mnist_dim, y1 + self.mnist_dim
        full_img = self.__draw_shape(full_img, x1, y1, x2, y2, digit_mask, digit_color)
        return full_img

    def __add_texture(self, full_img, texture_text, texture_color):
        sep = 6
        for x in np.arange(sep, self.biased_mnist_dim, self.biased_mnist_dim / sep):
            for y in np.arange(sep, self.biased_mnist_dim, self.biased_mnist_dim / sep):
                cv2.putText(full_img, texture_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=texture_color)
        return full_img

    def generate(self, save_dir, rewrite=False):
        # First load the configuration of all of the factors for all of the samples
        self.__init()
        sampled_factors = self.create_and_load_sampled_factors(save_dir, rewrite=rewrite)

        # Now let us generate the images
        for sample_ix, (digit_img, digit_lbl, factor) in enumerate(zip(self.images, self.labels, sampled_factors)):
            full_img = self.__create_new_image(factor['bg_color'])
            digit_position = self.grid_id_to_xy[factor['digit_cell_number']]
            digit_mask = self.__get_fg_mask(digit_img)
            full_img = self.__draw_distractor_shapes(full_img, factor['distractor'], factor['distractor_color'],
                                                     factor['digit_cell_number'])
            full_img = self.__add_texture(full_img, factor['texture'], factor['texture_color'])

            full_img = self.__draw_digit(full_img, digit_mask, factor['digit_color'], digit_position)
            img_save_dir = os.path.join(save_dir, self.filename)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            img_save_file = os.path.join(img_save_dir, str(factor['index']) + '.jpg')
            cv2.imwrite(img_save_file, full_img)

            p_save = np.random.random(1)
            if p_save < 0.01:
                sample_save_dir = img_save_dir + f"_samples_{digit_lbl}"
                if not os.path.exists(sample_save_dir):
                    os.makedirs(sample_save_dir)
                sample_save_file = os.path.join(sample_save_dir, str(factor['index']) + '.jpg')
                cv2.imwrite(sample_save_file, full_img)

            print(f"saved to {img_save_file}")


class GroupUtils():
    def __init__(self, num_classes=10, use_majority_minority_grouping=False):
        fv_obj = FactorToValues()
        self.factor_to_values = fv_obj.factor_to_values
        self.factor_value_to_ix = fv_obj.factor_value_to_ix
        self.num_classes = num_classes
        self.group_name_to_ix = {}
        self.maj_min_group_name_to_ix = {}
        self.max_group_ix = 0
        self.max_maj_min_group_ix = 0
        self.all_factor_unique_groups = self.__prepare_all_factor_unique_groups()
        self.use_majority_minority_grouping = use_majority_minority_grouping

    def __prepare_all_factor_unique_groups(self):
        factors = list(self.factor_to_values.keys())
        length = len(factors)
        combos = []

        # for i in np.arange(1, length + 1):
        for i in [1, length]:
            _combos = list(itertools.combinations(factors, i))
            combos += _combos

        return combos

    def to_group_ix_and_name(self, bias_variable_names, curr_factor_to_val):
        group_name_parts = []

        # Assume that if the label is same as the index of the factor val, then it is a majority group, else it is a minority group
        maj_min_group_name_parts = []

        lbl = curr_factor_to_val['label']

        # Go through all of the bias variables, to come up with the group name
        for ix, bias_name in enumerate(bias_variable_names):
            if bias_name == 'distractor':
                bias_val = str(curr_factor_to_val['primary_distractor'])
                bias_val_ix = self.factor_value_to_ix[bias_name][bias_val]
            elif bias_name == 'label':
                bias_val = lbl
                bias_val_ix = lbl
            else:
                bias_val = str(curr_factor_to_val[bias_name])
                bias_val_ix = self.factor_value_to_ix[bias_name][bias_val]
            maj_min = 'minority'
            if bias_val_ix == lbl:
                maj_min = 'majority'  # There is no majority/minority for lbl, so we just use 'majority' for label
            # group_ix += self.num_classes ** ix * bias_val_ix
            group_name_parts.append(f'{bias_name}_{bias_val}')
            # maj_min_group_ix += 2 ** ix * (0 if maj_min == 'majority' else 1)
            maj_min_group_name_parts.append(f'{bias_name}_{maj_min}')

        group_name = '+'.join(group_name_parts)
        maj_min_group_name = '+'.join(maj_min_group_name_parts)
        if self.use_majority_minority_grouping:
            group_name = maj_min_group_name

        if group_name not in self.group_name_to_ix:
            self.group_name_to_ix[group_name] = self.max_group_ix
            self.max_group_ix += 1
        if maj_min_group_name not in self.maj_min_group_name_to_ix:
            self.maj_min_group_name_to_ix[maj_min_group_name] = self.max_maj_min_group_ix
            self.max_maj_min_group_ix += 1
        group_ix = self.group_name_to_ix[group_name]
        maj_min_group_ix = self.maj_min_group_name_to_ix[maj_min_group_name]
        return group_ix, group_name, maj_min_group_ix, maj_min_group_name


class BiasedMNISTDataset(Dataset):
    """
    Directory structure:
    train and val images are present inside ${data_dir}/p_bias_{proba}/trainval
    train and val jsons are: ${data_dir}/p_bias_{proba}/trainval.json
    test images is inside ${data_dir}/test
    test json is inside ${data_dir}/test.json
    """

    def __init__(self, data_dir, p_bias, split, bias_variable_names, use_majority_minority_grouping):
        super(BiasedMNISTDataset, self).__init__()
        self.split = split
        self.use_majority_minority_grouping = use_majority_minority_grouping
        if 'train' in split or 'val' in split:
            self.images_dir = os.path.join(data_dir, f'p_bias_{p_bias}', 'trainval')
            self.factors_data = json.load(open(os.path.join(data_dir, f'p_bias_{p_bias}', 'trainval.json')))
        else:
            self.images_dir = os.path.join(data_dir, 'test')
            self.factors_data = json.load(open(os.path.join(data_dir, 'test.json')))

        self.num_classes = 10
        self.bias_variable_names = bias_variable_names
        self.main_group_utils = GroupUtils(use_majority_minority_grouping=self.use_majority_minority_grouping)
        self.prepare_dataset()
        if use_majority_minority_grouping:
            self.num_groups = self.main_group_utils.max_maj_min_group_ix
        else:
            self.num_groups = self.main_group_utils.max_group_ix

    def __len__(self):
        return len(self.factors_data)

    def prepare_dataset(self):
        # Pre-loading the data since grouping takes time
        self.data_items = {}
        for index, curr_factor_to_val in enumerate(self.factors_data):
            curr_factor_to_val = self.factors_data[index]
            group_ix, group_name, maj_min_group_ix, maj_min_group_name = self.main_group_utils.to_group_ix_and_name(
                self.bias_variable_names,
                curr_factor_to_val)
            if self.use_majority_minority_grouping:
                group_ix = maj_min_group_ix
                group_name = maj_min_group_name
            item_data = {
                'y': curr_factor_to_val['label'],
                'dataset_ix': index,
                'group_ix': group_ix,
                'group_name': group_name,
                'maj_min_group_ix': maj_min_group_ix,
                'maj_min_group_name': maj_min_group_name
            }

            # Let us create groups for each of the individual factors
            group_utils = GroupUtils()
            for factor in group_utils.factor_to_values:
                _grp_ix, _grp_name, _maj_min_grp_ix, _maj_min_grp_name = group_utils.to_group_ix_and_name(
                    [factor],
                    curr_factor_to_val)
                item_data[f'{factor}_maj_min_group_name'] = _maj_min_grp_name
                item_data[f'{factor}_maj_min_group_ix'] = _maj_min_grp_ix

            # Let us create groups for other grouping variables
            group_utils = GroupUtils()
            for afug in group_utils.all_factor_unique_groups:
                unq_grp_name = '+'.join(afug)
                _grp_ix, _grp_name, _maj_min_grp_ix, _maj_min_grp_name = group_utils.to_group_ix_and_name(afug,
                                                                                                          curr_factor_to_val)
                item_data[f'{unq_grp_name}_maj_min_group_ix'] = _maj_min_grp_ix
                item_data[f'{unq_grp_name}_maj_min_group_name'] = _maj_min_grp_name
            self.data_items[index] = item_data
        # logging.getLogger().info("self.maj_min_group_name_to_ix")
        # logging.getLogger().info(json.dumps(self.main_group_utils.maj_min_group_name_to_ix, indent=4, sort_keys=True))

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.images_dir, f'{index}.jpg'))
        img = torch.FloatTensor(img)
        img = img.permute((2, 0, 1))

        item_data = self.data_items[index]
        item_data['x'] = img
        return item_data


def create_biased_mnist_datasets(data_dir, bias_variable_names, bias_proba, use_majority_minority_grouping):
    train_ixs = json.load(open(os.path.join(data_dir, 'train_ixs.json')))
    train_set = BiasedMNISTDataset(data_dir, bias_proba, 'train', bias_variable_names, use_majority_minority_grouping)
    num_groups = train_set.num_groups
    train_set = Subset(train_set, train_ixs)

    val_ixs = json.load(open(os.path.join(data_dir, 'val_ixs.json')))
    val_set = BiasedMNISTDataset(data_dir, bias_proba, 'val', bias_variable_names, use_majority_minority_grouping)
    val_set = Subset(val_set, val_ixs)

    balanced_val_set = BiasedMNISTDataset(data_dir, 0.1, 'val', bias_variable_names, use_majority_minority_grouping)
    balanced_val_set = Subset(balanced_val_set, val_ixs)

    test_set = BiasedMNISTDataset(data_dir, None, 'test', bias_variable_names, use_majority_minority_grouping)
    return train_set, val_set, balanced_val_set, test_set, num_groups


def create_biased_mnist_dataloaders(option):
    train_set, val_set, balanced_val_set, test_set, num_groups = create_biased_mnist_datasets(option.data_dir,
                                                                                              option.bias_variables,
                                                                                              option.bias_proba,
                                                                                              option.use_majority_minority_grouping)
    logging.getLogger().info(f"Setting the num_groups to {num_groups}")
    option.num_groups = num_groups
    option.bias_variable_dims = num_groups
    option.bias_model_hid_dims = 32
    option.bias_predictor_hid_dims = 32
    option.num_bias_classes = num_groups

    train_loader = DataLoader(train_set, batch_size=option.batch_size, shuffle=True, num_workers=option.num_workers,
                              collate_fn=dict_collate_fn())
    val_loader = DataLoader(val_set, batch_size=option.batch_size, shuffle=False, num_workers=option.num_workers,
                            collate_fn=dict_collate_fn())
    balanced_val_loader = DataLoader(balanced_val_set, batch_size=option.batch_size, shuffle=False,
                                     num_workers=option.num_workers, collate_fn=dict_collate_fn())
    test_loader = DataLoader(test_set, batch_size=option.batch_size, shuffle=False, num_workers=option.num_workers,
                             collate_fn=dict_collate_fn())
    return {
        'Train': train_loader,
        'Test': {
            'Train': train_loader,
            'Val': val_loader,
            'Balanced Val': balanced_val_loader,
            'Test': test_loader
        }
    }


def generate_biased_mnist_dataset():
    start = time.time()
    ####################################################################################################################
    # Carve out train vs val indices
    ####################################################################################################################
    total_indices = 60000
    indices = np.arange(0, total_indices)
    np.random.shuffle(indices)
    train_indices = indices[0:50000].tolist()
    val_indices = indices[50000:total_indices].tolist()
    json.dump(train_indices, open(os.path.join(biased_mnist_dir, 'train_ixs.json'), 'w'))
    json.dump(val_indices, open(os.path.join(biased_mnist_dir, 'val_ixs.json'), 'w'))
    bias_factors = ['digit_color', 'bg_color', 'distractor', 'distractor_color', 'texture', 'texture_color',
                    'digit_cell_number']

    ###################################################################################################################
    # Generate trainval sets with different degrees of biases
    ###################################################################################################################
    for p_bias in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # for p_bias in [0.1, 0.7, 0.9]:
        # for p_bias in [0.7]:
        save_dir = os.path.join(biased_mnist_dir, f'p_bias_{p_bias}')
        print(f"Saving to {save_dir}")
        bias_probas = {}
        for f in bias_factors:
            bias_probas[f] = p_bias

        factory = BiasedMNISTGenerator(original_mnist_dir, 'train', bias_probabilities=bias_probas)
        factory.generate(save_dir, rewrite=True)

    ####################################################################################################################
    # Generate an unbiased test split
    ####################################################################################################################
    p_bias = 0.1
    bias_probas = {}
    for f in bias_factors:
        bias_probas[f] = p_bias

    factory = BiasedMNISTGenerator(original_mnist_dir, 'test', bias_probabilities=bias_probas)
    factory.generate(biased_mnist_dir, rewrite=True)

    print(f"Completed in {time.time() - start}")


def test_grouping():
    ## Test grouping
    curr_factor = {
        "bg_color": [
            63,
            0,
            31
        ],
        "digit_cell_number": 5,
        "digit_color": [
            128,
            255,
            0
        ],
        "distractor": [
            "plus",
            "kite",
            "circle",
            "obtuse_triangle",
            "right_triangle",
            "kite",
            "square",
            "square",
            "plus"
        ],
        "distractor_color": [
            0,
            255,
            255
        ],
        "index": 0,
        "label": 7,
        "primary_distractor": "random",
        "texture": "x",
        "texture_color": [
            255,
            255,
            0
        ]
    }
    group_utils = GroupUtils()
    a = group_utils.to_group_ix_and_name(['digit_color',
                                          'distractor_color',
                                          'bg_color',
                                          'distractor',
                                          'texture',
                                          'texture_color',
                                          'digit_cell_number'],
                                         curr_factor)
    print(a)


if __name__ == "__main__":
    # # Generate the different MNIST datasets
    original_mnist_dir = option.ROOT + '/MNIST'
    biased_mnist_dir = option.ROOT + '/biased_mnist_v1'
    if not os.path.exists(biased_mnist_dir):
        os.makedirs(biased_mnist_dir)
    generate_biased_mnist_dataset()

    # setattr(option, 'data_dir', biased_mnist_dir)
    # setattr(option, 'bias_proba', 0.5)
    # setattr(option, 'bias_variables', ['fg_color', 'bg_color'])
    # setattr(option, 'batch_size', 128)
    # setattr(option, 'num_workers', 0)
    # loaders = create_biased_mnist_dataloaders(option)
    # train_loader = loaders['Train']
    # for batch in train_loader:
    #     for key in batch:
    #         if key != 'x':
    #             print(f"key: {key}")
    #             print(batch[key])
    #     break

    # copy the samples
