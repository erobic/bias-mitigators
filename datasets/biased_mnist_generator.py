from multiprocessing import Process
from torchvision import datasets, transforms

import numpy as np
import copy
import os, json
import itertools
import cv2
from emnist import extract_training_samples, extract_test_samples
import colorsys
from json import JSONEncoder
from datasets.biased_mnist_generator_utils import *
import yaml
import argparse

NUM_CLASSES = 10


class BiasedMNISTGenerator():
    """
    Bias factors: digit_color, digit_scale, texture, letter are correlated with the digit class
    Bias factor: texture_color is correlated with the texture class
    Also supports class imbalance
    """

    def __init__(self, original_mnist_dir, textures_dir, split, bias_config, cell_dim=32):
        """

        :param original_mnist_dir: Directory where the MNIST images are stored
        :param textures_dir: Directory where the texture images are stored. Refer to get_textures() function in texture_utils.py
        :param split: train, val or test
        :param bias_config: For each bias type, provide 'enabled' and 'p_bias' e.g., 'digit_color': {'enabled': true, 'p_bias': 0.996}
        bias_config also needs to contain 'num_cells' (default set to 5) and class_imbalance_ratio (which is the ratio
         between # samples in most frequent class to least frequent class)
        """
        self.split = split
        self.original_mnist_dir = original_mnist_dir
        self.bias_config = bias_config
        self.max_letter_cnt = 3

        self.num_classes = 10
        self.num_cells = bias_config['num_cells']
        self.image_dim = cell_dim * self.num_cells
        self.textures_dir = textures_dir
        self.class_imbalance_ratio = bias_config['class_imbalance_ratio']

        # BIAS VARIABLE NAMES
        self.BIAS_DIGIT_COLOR = 'digit_color'
        self.BIAS_DIGIT_SCALE = 'digit_scale'
        self.BIAS_TEXTURE = 'texture'
        self.BIAS_TEXTURE_COLOR = 'texture_color'
        self.BIAS_NATURAL_TEXTURE = 'natural_texture'
        self.BIAS_DIGIT_POSITION = 'digit_position'
        self.BIAS_LETTER = 'letter'
        self.BIAS_LETTER_COLOR = 'letter_color'

        self.load_mnist()
        self.mnist_dim = 32
        self.letter_dim = 28

    def get_p_bias(self, bias_name):
        """
        Returns the probability of association
        :param bias_name:
        :return:
        """
        return self.bias_config[bias_name]['p_bias']

    def is_bias_enabled(self, bias_name):
        """
        Returns true only if the factor acts as a bias variable
        :param bias_name:
        :return:
        """
        return self.bias_config[bias_name]['enabled']

    def load_mnist(self):
        if self.split == 'train':
            self.original_dataset = datasets.MNIST(self.original_mnist_dir, train=True, download=True,
                                                   transform=transforms.ToTensor())
            self.filename = 'trainval'
        else:
            self.original_dataset = datasets.MNIST(self.original_mnist_dir, train=False, download=True,
                                                   transform=transforms.ToTensor())
            self.filename = self.split

        self.mnist_images, self.mnist_digits = dataset_to_xy(self.original_dataset)
        if self.textures_dir is not None:
            self.texture_to_images = load_texture_images(self.textures_dir)

        # Perform class imbalanced sampling if necessary
        if self.class_imbalance_ratio is not None:
            item_ixs = sample_long_tailed(self.mnist_digits, self.class_imbalance_ratio, num_classes=self.num_classes)
            self.mnist_images = self.mnist_images[item_ixs]
            self.mnist_digits = self.mnist_digits[item_ixs]

    def sample_digit_colors(self, class_ixs):
        bias_name = self.BIAS_DIGIT_COLOR
        if self.is_bias_enabled(bias_name):

            if self.bias_config[bias_name]['type'] == 'perturbed':
                digit_color_ixs, digit_hues = sample_conditional_biased_values(get_digit_hues(), class_ixs,
                                                                               self.get_p_bias(bias_name))
                digit_colors = perturb_saturation_and_values(digit_hues)
            else:
                digit_color_ixs, digit_colors = sample_conditional_biased_values(get_digit_colors(), class_ixs,
                                                                                 self.get_p_bias(bias_name))
        else:
            digit_color_ixs, digit_colors = [0] * len(class_ixs), [get_default_digit_color()] * len(class_ixs)
        return digit_color_ixs, digit_colors

    def sample_digit_scales(self, class_ixs):
        bias_name = self.BIAS_DIGIT_SCALE
        if self.is_bias_enabled(bias_name):
            digit_scale_ixs, digit_scales = sample_conditional_biased_values(get_digit_scales(self.num_cells),
                                                                             class_ixs,
                                                                             self.get_p_bias(bias_name))
        else:
            digit_scale_ixs, digit_scales = [0] * len(class_ixs), [get_default_digit_scale()] * len(class_ixs)
        return digit_scale_ixs, digit_scales

    def sample_digit_positions(self, class_ixs, digit_scale_ixs, digit_scales):
        bias_name = self.BIAS_DIGIT_POSITION
        if self.is_bias_enabled(bias_name):
            # TODO: handle this as well
            digit_position_ixs, digit_positions = sample_biased_digit_positions(digit_scale_ixs,
                                                                                get_scale_ix_to_digit_positions(),
                                                                                class_ixs,
                                                                                self.get_p_bias(bias_name))
        else:
            digit_position_ixs, digit_positions = [0] * len(class_ixs), get_center_positions(digit_scales,
                                                                                             num_cells=self.num_cells)
        return digit_scale_ixs, digit_positions

    # Texture
    def sample_textures(self, class_ixs):
        bias_name = self.BIAS_TEXTURE
        if self.is_bias_enabled(bias_name):
            texture_ixs, textures = sample_conditional_biased_values(get_textures(), class_ixs,
                                                                     self.get_p_bias(bias_name))
        else:
            texture_ixs, textures = [0] * len(class_ixs), [get_default_texture()] * len(class_ixs)
        # texture_file_rands, x1s, x2s, y1s, y2s = sample_texture_crops(len(texture_ixs), self.split, self.image_dim)
        return texture_ixs, textures

    def sample_texture_colors(self, digit_ixs, texture_ixs):
        bias_name = self.BIAS_TEXTURE_COLOR
        if self.is_bias_enabled(bias_name):
            texture_color_ixs, texture_colors = sample_conditional_biased_values(get_texture_colors(),
                                                                                 texture_ixs,
                                                                                 self.get_p_bias(bias_name))
        else:
            texture_color_ixs, texture_colors = [0] * len(texture_ixs), [get_default_texture_color()] * len(texture_ixs)
        return texture_color_ixs, texture_colors

    # Natural image as texture
    def sample_natural_textures(self, class_ixs):
        bias_name = self.BIAS_NATURAL_TEXTURE
        if self.is_bias_enabled(bias_name):
            texture_ixs, textures = sample_conditional_biased_values(get_natural_textures(), class_ixs,
                                                                     self.get_p_bias(bias_name))
        else:
            texture_ixs, textures = [0] * len(class_ixs), [get_default_natural_texture()] * len(class_ixs)
        texture_file_rands, x1s, x2s, y1s, y2s = sample_texture_crops(len(texture_ixs), self.split, self.image_dim)
        return texture_ixs, textures, texture_file_rands, x1s, x2s, y1s, y2s

    def sample_natural_texture_colors(self, digit_ixs, texture_ixs):
        bias_name = self.BIAS_TEXTURE_COLOR
        if self.is_bias_enabled(bias_name):
            texture_color_ixs, texture_colors = sample_conditional_biased_values(get_texture_colors(),
                                                                                 texture_ixs,
                                                                                 self.get_p_bias(bias_name))
        else:
            texture_color_ixs, texture_colors = [0] * len(texture_ixs), [get_default_texture_color()] * len(texture_ixs)
        return texture_color_ixs, texture_colors

    def sample_letters(self, digit_ixs):
        bias_name = self.BIAS_LETTER
        p_bias = self.get_p_bias(bias_name)
        letter_ixs, letters = sample_conditional_biased_values(get_letters(),
                                                               digit_ixs,
                                                               p_bias)
        return letter_ixs, letters

    def sample_letter_colors(self, digit_ixs, letter_ixs):
        bias_name = self.BIAS_LETTER_COLOR
        p_bias = self.get_p_bias(bias_name)
        letter_color_ixs, letter_colors = sample_conditional_biased_values(get_letter_colors(),
                                                                           letter_ixs,
                                                                           p_bias)
        return letter_color_ixs, letter_colors

    def sample_attributes(self):
        """
        For each digit, we sample all the factors
        :return:
        """
        digit_ixs = self.mnist_digits
        attrs = {'digits': digit_ixs}

        texture_ixs, textures = self.sample_textures(digit_ixs)
        attrs['texture_ixs'] = texture_ixs
        attrs['textures'] = textures

        if self.is_bias_enabled(self.BIAS_NATURAL_TEXTURE):
            natural_texture_ixs, natural_textures, natural_texture_file_rands, x1s, x2s, y1s, y2s = self.sample_natural_textures(
                digit_ixs)
            attrs['natural_texture_ixs'] = natural_texture_ixs
            attrs['natural_textures'] = natural_textures
            attrs['natural_texture_file_rands'] = natural_texture_file_rands
            attrs['natural_texture_x1s'] = x1s
            attrs['natural_texture_x2s'] = x2s
            attrs['natural_texture_y1s'] = y1s
            attrs['natural_texture_y2s'] = y2s

        texture_color_ixs, texture_colors = self.sample_texture_colors(digit_ixs, texture_ixs)
        attrs['texture_color_ixs'] = texture_color_ixs
        attrs['texture_colors'] = texture_colors

        digit_color_ixs, digit_colors = self.sample_digit_colors(digit_ixs)
        attrs['digit_color_ixs'] = digit_color_ixs
        attrs['digit_colors'] = digit_colors

        digit_scale_ixs, digit_scales = self.sample_digit_scales(digit_ixs)
        attrs['digit_scale_ixs'] = digit_scale_ixs
        attrs['digit_scales'] = digit_scales

        digit_position_ixs, digit_positions = self.sample_digit_positions(digit_ixs, digit_scale_ixs, digit_scales)
        attrs['digit_position_ixs'] = digit_position_ixs
        attrs['digit_positions'] = digit_positions

        if self.is_bias_enabled(self.BIAS_LETTER):
            letter_ixs, letters = self.sample_letters(digit_ixs)
            attrs['letter_ixs'] = letter_ixs
            attrs['letters'] = letters

        if self.is_bias_enabled(self.BIAS_LETTER_COLOR):
            letter_color_ixs, letter_colors = self.sample_letter_colors(digit_ixs, letter_ixs)
            attrs['letter_color_ixs'] = letter_color_ixs
            attrs['letter_colors'] = letter_colors

        return attrs

    def apply_color_by_threshold(self, src_img, target_rgb, threshold=0):
        if len(src_img.shape) == 2:
            grayscale = src_img
        else:
            # First convert to grayscale
            grayscale = src_img.mean(axis=2)
        grayscale = np.expand_dims(grayscale, 2).repeat(3, axis=2)
        colored_img = np.copy(grayscale)
        # Then apply colors to each channel
        for ch in [0, 1, 2]:
            colored_img[:, :, ch] = np.where(grayscale[:, :, ch] > threshold, target_rgb[2 - ch],
                                             grayscale[:, :, ch])
        return colored_img

    def apply_color(self, src_img, target_rgb):
        if len(src_img.shape) == 2:
            grayscale = src_img
        else:
            # First convert to grayscale
            grayscale = src_img.mean(axis=2)
        grayscale = np.expand_dims(grayscale, 2).repeat(3, axis=2)
        colored_img = np.copy(grayscale)
        # Then apply colors to each channel
        for ch in [0, 1, 2]:
            colored_img[:, :, ch] = (grayscale[:, :, ch] / 255) * target_rgb[2 - ch]
        return colored_img

    def add_sub_img(self, src_img, target_img, mask_img, row1, col1, row2, col2, blend=False):
        mask = mask_img[:, :, 0] + mask_img[:, :, 1] + mask_img[:, :, 2]
        for channel in [0, 1, 2]:
            target_sub_img = target_img[row1:row2, col1:col2, channel]
            if blend:
                blended_sub_img = (src_img[:, :, channel] + target_sub_img) / 2
            else:
                blended_sub_img = src_img[:, :, channel]
            target_img[row1:row2, col1:col2, channel] = np.where(mask == 0, target_sub_img, blended_sub_img)

        return target_img

    def apply_texture(self, img, attributes, ix, sample_labels):
        texture_cnt = np.random.randint(6, 18, size=1)
        texture_text = attributes['textures'][ix]
        texture_color = attributes['texture_colors'][ix]
        for x in np.arange(0, self.image_dim, self.image_dim / texture_cnt):
            for y in np.arange(0, self.image_dim, self.image_dim / texture_cnt):
                cv2.putText(img, texture_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=texture_color)
        sample_labels['texture'] = attributes['textures'][ix]
        sample_labels['texture_ix'] = attributes['texture_ixs'][ix]
        sample_labels['texture_color'] = texture_color
        sample_labels['texture_color_ix'] = attributes['texture_color_ixs'][ix]

        return img

    def apply_natural_texture(self, attributes, ix, sample_labels):
        texture = attributes['natural_textures'][ix]
        texture_img_ix = int(len(self.texture_to_images[texture]) * attributes['natural_texture_file_rands'][ix])
        texture_full_img = copy.deepcopy(self.texture_to_images[texture][texture_img_ix])

        # Texture hue
        texture_color = attributes['texture_colors'][ix]

        # Crop the texture image
        cropped_texture_img = texture_full_img[
                              attributes['natural_texture_x1s'][ix]:attributes['natural_texture_x1s'][
                                                                        ix] + self.image_dim,
                              attributes['natural_texture_y1s'][ix]:attributes['natural_texture_y1s'][
                                                                        ix] + self.image_dim]

        # Change the hue of the texture image
        cropped_texture_img = self.apply_color(cropped_texture_img, texture_color)

        sample_labels['natural_texture'] = attributes['natural_textures'][ix]
        sample_labels['natural_texture_ix'] = attributes['natural_texture_ixs'][ix]
        sample_labels['natural_texture_color'] = texture_color
        sample_labels['natural_texture_color_ix'] = attributes['natural_texture_color_ixs'][ix]

        return cropped_texture_img

    def apply_mnist(self, img, attributes, ix, sample_labels, blend=False):
        mnist_img = self.mnist_images[ix]
        mnist_dim = mnist_img.shape[0]
        mnist_position = attributes['digit_positions'][ix]

        # Determine size and position of the digit
        scale = attributes['digit_scales'][ix]
        row1, col1 = int(mnist_position[0] * mnist_dim), int(mnist_position[1] * mnist_dim)
        row2, col2 = int(row1 + mnist_dim * scale), int(col1 + mnist_dim * scale)
        mnist_img = cv2.resize(mnist_img, (row2 - row1, col2 - col1), interpolation=cv2.INTER_NEAREST)

        # Change color
        color = np.asarray(attributes['digit_colors'][ix])
        colored_img = self.apply_color_by_threshold(mnist_img, color, threshold=20)

        self.add_sub_img(colored_img, img, mnist_img, row1, col1, row2, col2, blend=blend)
        sample_labels['digit_color'] = attributes['digit_colors'][ix]
        sample_labels['digit_color_ix'] = attributes['digit_color_ixs'][ix]

        sample_labels['digit_scale'] = attributes['digit_scales'][ix]
        sample_labels['digit_scale_ix'] = attributes['digit_scale_ixs'][ix]

        sample_labels['digit_position'] = (row1, col1, row2, col2)
        sample_labels['digit_position_ix'] = attributes['digit_position_ixs'][ix]

        return img

    def apply_co_occurring_letters(self, img, attributes, ix, sample_labels):
        def overlaps_with_digit(row_cell, col_cell):
            row1, col1, row2, col2 = sample_labels['digit_position']
            mnist_dim = self.mnist_images[ix].shape[0]
            row = row_cell * mnist_dim + mnist_dim / 2
            col = col_cell * mnist_dim + mnist_dim / 2
            pad = 0
            return (row1 + pad) <= row <= (row2 - pad) and (col1 + pad) <= col <= (col2 - pad)

        if not hasattr(self, 'letter_to_images'):
            self.letter_to_images = load_letter_ix_to_images(self.split)

        letter_ix = attributes['letter_ixs'][ix]
        candidate_letter_positions = []

        for row_cell in np.arange(0, self.num_cells):
            for col_cell in np.arange(0, self.num_cells):
                if not overlaps_with_digit(row_cell, col_cell):
                    candidate_letter_positions.append((row_cell, col_cell))

        if len(candidate_letter_positions) > 0:
            position_ixs = np.random.choice(np.arange(0, len(candidate_letter_positions)), size=self.max_letter_cnt,
                                            replace=False)
            candidate_letter_positions = [candidate_letter_positions[ix] for ix in position_ixs]
        else:
            print("No candidate letters!")
        letter_position_ixs = np.arange(0, len(candidate_letter_positions))
        letter_color = attributes['letter_colors'][ix]
        for _ix, letter_pos_ix in enumerate(letter_position_ixs):
            letter_pos = candidate_letter_positions[letter_pos_ix]
            row_cell, col_cell = letter_pos[0], letter_pos[1]
            letter_img_ix = np.random.randint(0, len(self.letter_to_images[letter_ix]))
            letter_r1, letter_c1 = row_cell * self.mnist_dim, col_cell * self.mnist_dim
            letter_r2, letter_c2 = letter_r1 + self.letter_dim, letter_c1 + self.letter_dim

            # Draw letter
            letter_img = self.letter_to_images[letter_ix][letter_img_ix]
            letter_img = self.apply_color(letter_img, letter_color)
            self.add_sub_img(letter_img, img, letter_img, letter_r1, letter_c1, letter_r2, letter_c2)

        for attr_key in ['letter_ix', 'letter', 'letter_color_ix', 'letter_color']:
            sample_labels[attr_key] = attributes[f'{attr_key}s'][ix]

        return img

    def generate(self, save_dir):
        """
        Has all the logic to actually generate the image
        :param save_dir:
        :param rewrite:
        :return:
        """
        # Prepare the directory where images will be saved
        attributes = self.sample_attributes()
        img_save_dir = os.path.join(save_dir, self.filename)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        # Now let us generate the images
        sample_factors = []
        for ix, (mnist_img, digit) in enumerate(zip(self.mnist_images, self.mnist_digits)):
            _sample_labels = {'index': ix, 'digit': digit}
            # img = self.apply_natural_texture(attributes, ix, curr_factors)
            img = np.zeros((self.image_dim, self.image_dim, 3))
            if self.is_bias_enabled(self.BIAS_TEXTURE):
                blend = True

            img = self.apply_mnist(img, attributes, ix, _sample_labels, blend=blend)
            img = self.apply_texture(img, attributes, ix, _sample_labels)
            blend = False

            # Add co-occurring shapes
            if self.is_bias_enabled(self.BIAS_LETTER):
                img = self.apply_co_occurring_letters(img, attributes, ix, _sample_labels)

            # img = self.apply_mnist(img, attributes, ix, _sample_labels, blend=blend)
            # Save the images
            img_save_file = os.path.join(img_save_dir, str(ix) + '.jpg')
            cv2.imwrite(img_save_file, img)

            # Save some samples for visualization
            self._save_for_visualization(img, img_save_dir, digit, ix, _sample_labels)

            sample_factors.append(_sample_labels)

        with open(os.path.join(save_dir, self.filename + '.json'), 'w') as f:
            json.dump(sample_factors, f, indent=4, sort_keys=True, cls=NpEncoder)
        self.save_dir = save_dir

    def _save_for_visualization(self, img, img_save_dir, digit, ix, curr_factors):
        # Save some samples for visualization
        p_save = np.random.random(1)
        if p_save < 0.01:
            sample_save_dir = img_save_dir + f"_samples_{digit}"
            if not os.path.exists(sample_save_dir):
                os.makedirs(sample_save_dir)
            sample_save_file = os.path.join(sample_save_dir, str(ix) + '.jpg')
            cv2.imwrite(sample_save_file, img)


def get_p_bias_config(digit_color_enabled=False, digit_color_p_bias=0.1,
                      digit_color_type='discrete',
                      digit_scale_enabled=False, digit_scale_p_bias=0.1,
                      digit_position_enabled=False, digit_position_p_bias=0.1,
                      texture_enabled=False, texture_p_bias=0.1,
                      texture_color_enabled=False, texture_color_p_bias=0.1,
                      letter_enabled=False, letter_p_bias=0.1,
                      letter_color_enabled=False, letter_color_p_bias=0.1,
                      num_cells=5, class_imbalance_ratio=None):
    return {
        'digit_color': {
            'enabled': digit_color_enabled,
            'p_bias': to_list(digit_color_p_bias),
            'type': digit_color_type
        },
        'digit_scale': {
            'enabled': digit_scale_enabled,
            'p_bias': to_list(digit_scale_p_bias),
        },
        'digit_position': {
            'enabled': digit_position_enabled,
            'p_bias': to_list(digit_position_p_bias),
        },
        'texture': {
            'enabled': texture_enabled,
            'p_bias': to_list(texture_p_bias),
        },
        'texture_color': {
            'enabled': texture_color_enabled,
            'p_bias': to_list(texture_color_p_bias)
        },
        'letter': {
            'enabled': letter_enabled,
            'p_bias': to_list(letter_p_bias)
        },
        'letter_color': {
            'enabled': letter_color_enabled,
            'p_bias': to_list(letter_color_p_bias)
        },
        'natural_texture': {
            'enabled': False,
            'p_bias': False
        },
        'num_cells': num_cells,
        'class_imbalance_ratio': class_imbalance_ratio
    }


def generate_train_val_ixs(biased_mnist_dir, replace=False):
    total_indices = 60000
    indices = np.arange(0, total_indices)
    np.random.shuffle(indices)
    train_indices = indices[0:50000].tolist()
    val_indices = indices[50000:total_indices].tolist()
    train_ixs_path = os.path.join(biased_mnist_dir, 'train_ixs.json')
    if replace or not os.path.exists(train_ixs_path):
        json.dump(train_indices, open(os.path.join(biased_mnist_dir, 'train_ixs.json'), 'w'))

    val_ixs_path = os.path.join(biased_mnist_dir, 'val_ixs.json')
    if replace or not os.path.exists(val_ixs_path):
        json.dump(val_indices, open(os.path.join(biased_mnist_dir, 'val_ixs.json'), 'w'))


def generate_splits(bias_split_name, original_mnist_dir, biased_mnist_dir, textures_dir, bias_config,
                    test_bias_config, suffix=None, generate_test_set=False):
    print(f"Generating {f'{bias_split_name}{suffix}'}")

    save_dir = os.path.join(biased_mnist_dir, f'{bias_split_name}{suffix}')

    # Generate the trainval set (train_ixs.json and val_ixs.json are used to distinguish them)
    generator = BiasedMNISTGenerator(original_mnist_dir, textures_dir, split='train', bias_config=bias_config)
    generator.generate(save_dir)

    # Generate the test set
    if generate_test_set:
        test_generator = BiasedMNISTGenerator(original_mnist_dir, textures_dir, split='test',
                                              bias_config=test_bias_config)
        test_save_dir = os.path.join(biased_mnist_dir, bias_split_name)
        test_generator.generate(test_save_dir)

    print(f"Generated {f'{bias_split_name}{suffix}'}")
    return generator


def replace_p_bias(bias_config, p_bias):
    for key in bias_config:

        if isinstance(bias_config[key], dict) and 'p_bias' in list(bias_config[key].keys()) and bias_config[key][
            'enabled']:
            bias_config[key]['p_bias'] = p_bias
    return bias_config


def p_bias_to_list(bias_config):
    for key in bias_config:
        if isinstance(bias_config[key], dict) and 'p_bias' in list(bias_config[key].keys()):
            if isinstance(bias_config[key]['p_bias'], float):
                bias_config[key]['p_bias'] = [bias_config[key]['p_bias']] * NUM_CLASSES
    return bias_config


def main(cfg, p_bias, suffix, generate_test_set):
    bias_config = cfg['bias_config']
    biased_mnist_dir = cfg['biased_mnist_dir']
    bias_split_name = cfg['bias_split_name']
    mnist_dir = cfg['mnist_dir']
    textures_dir = cfg['textures_dir']

    generate_train_val_ixs(biased_mnist_dir, replace=False)  # Do this only once (common to all splits)
    if not os.path.exists(biased_mnist_dir):
        os.makedirs(biased_mnist_dir)

    if p_bias is not None:
        bias_config = replace_p_bias(bias_config, p_bias)
    bias_config = p_bias_to_list(bias_config)

    test_bias_config = replace_p_bias(copy.deepcopy(bias_config), 0.1)
    test_bias_config = p_bias_to_list(test_bias_config)

    generate_splits(bias_split_name, mnist_dir, biased_mnist_dir, textures_dir, bias_config=bias_config,
                    test_bias_config=test_bias_config, suffix=suffix, generate_test_set=generate_test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help="Dataset configuration file, specifying: bias_config, etc.")
    parser.add_argument('--p_bias', type=float,
                        help="If specified, then it will override the p_bias values specified in the config_file",
                        default=None)
    parser.add_argument('--suffix', type=str, help="If specified, then the bias split name will have this suffix",
                        default='')
    parser.add_argument('--generate_test_set', type=int, default=0,
                        help='Generates test set with p_bias=0.1 if set to true')
    args = parser.parse_args()
    with open(args.config_file) as f:
        dataset_config = yaml.safe_load(f)
    main(dataset_config, args.p_bias, args.suffix, args.generate_test_set)
