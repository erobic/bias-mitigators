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

NUM_CLASSES = 10


def preprocess_mnist(img, width=32, height=32):
    img = np.repeat(img, 3, axis=2)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) * 255
    return img


def dataset_to_xy(dataset, width=32, height=32):
    images, labels = [], []
    for img, label in dataset:
        img = np.asarray(img).transpose(1, 2, 0)
        img = preprocess_mnist(img, width, height)
        images.append(img)
        labels.append(label)
    return np.asarray(images), np.asarray(labels)


def get_non_biased_values_per_class(class_ix_to_factor_value, num_classes=10):
    """
    Assumes that each class co-occurs frequently with the factor value at the same index
    :return:
    """
    class_ix_to_non_biased_values = {}
    for class_ix in np.arange(0, num_classes):
        non_biased_vals = copy.deepcopy(class_ix_to_factor_value)
        del non_biased_vals[class_ix]  # Remove the biased value
        class_ix_to_non_biased_values[class_ix] = non_biased_vals
    return class_ix_to_non_biased_values


def get_digit_colors():
    digit_colors = [
        (200, 200, 200),
        (0, 150, 255),
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 235),
        (255, 140, 0),
        (155, 3, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
    ]
    return digit_colors


def get_digit_hues():
    digit_hues = [
        0,  # red
        20,  # brownish
        40,  # orange
        60,  # yellow
        90,  # light green
        140,  # green
        175,  # cyan
        220,  # blue
        270,  # purple,
        295,  # pink
    ]
    return digit_hues


def get_default_digit_color():
    return (255, 255, 255)


def get_letter_colors():
    return np.flip(get_digit_colors())


def apply_color(src_img, target_rgb):
    if len(src_img.shape) == 2:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    bgr = src_img.mean(axis=2)
    bgr = np.expand_dims(bgr, 2).repeat(3, axis=2)
    bgr[:, :, 0] *= target_rgb[2] / 255
    bgr[:, :, 1] *= target_rgb[1] / 255
    bgr[:, :, 2] *= target_rgb[0] / 255
    return bgr


def get_texture_colors():
    texture_colors = [(r / 1.5, g / 1.5, b / 1.5) for (r, g, b) in np.flip(get_digit_colors(), axis=0)]
    return texture_colors


def get_default_texture_color():
    return (0, 0, 0)


def perturb_saturation_and_values(hues, min_saturation=50, min_value=70):
    saturations = np.random.uniform(low=min_saturation, high=100, size=(len(hues)))
    values = np.random.uniform(low=min_value, high=100, size=(len(hues)))
    rgbs = np.asarray(
        [colorsys.hsv_to_rgb(h / 360, s / 100, v / 100) for (h, s, v) in zip(hues, saturations, values)]) * 255.
    return rgbs


def get_digit_scales(num_cells=5, pad_cells=1):
    """
    We define scale as the fraction of the image (in terms of number of cells in horizontal/vertical dims) occupied
    by the digits.
    :param num_cells:
    :param pad_cells:
    :return:
    """
    return np.linspace(1, num_cells - pad_cells, 10).tolist()


def get_default_digit_scale():
    return 1


def get_scale_ix_to_digit_positions(num_cells=5, pad_cells=0):
    """
    Different scales have different possible positions e.g., x1, y1 of a 3x3 digit needs to be at least (maxCellX - 3, maxCellY - 3)
    :param num_cells:
    :param pad_cells:
    :return:
    """
    scales = get_digit_scales(num_cells, pad_cells)
    scale_ix_to_digit_positions = {}
    for scale_ix, scale in enumerate(scales):
        scale_positions = []
        for row in np.arange(0, num_cells - np.ceil(scale) + 1):
            for col in np.arange(0, num_cells - np.ceil(scale) + 1):
                scale_positions.append((row, col))
        np.random.shuffle(scale_positions)
        scale_ix_to_digit_positions[scale_ix] = scale_positions
    return scale_ix_to_digit_positions


def sample_biased_digit_positions(digit_scale_ixs, scale_ix_to_positions, class_ixs, p_bias):
    """
    Returns the biased/canonical value of position for a given scale with probability p_bias, else chooses position at random
    :param digit_scale_ixs:
    :param scale_ix_to_positions: map from scale to potential positions
    :param p_bias:
    :return:
    """
    value_to_ix = {}
    max_value_to_ix = 0

    random_p = np.random.random(len(digit_scale_ixs))

    # Choose the values for each factor
    sampled_factor_ixs = []
    sampled_factors = []
    for ix, (scale_ix, cls_ix) in enumerate(zip(digit_scale_ixs, class_ixs)):
        if random_p[ix] <= p_bias[cls_ix]:
            # Assign the biased value for each factor
            sampled_factor = scale_ix_to_positions[scale_ix][0]
        else:
            # Sample uniformly from rest of the factors
            if len(scale_ix_to_positions[scale_ix]) > 1:
                arr = np.asarray(scale_ix_to_positions[scale_ix][1:])
                arr_ix = np.random.randint(0, len(arr))
                sampled_factor = arr[arr_ix]
            else:
                sampled_factor = scale_ix_to_positions[scale_ix][0]

        sampled_factors.append(sampled_factor)
        if str(sampled_factor) not in value_to_ix:
            value_to_ix[str(sampled_factor)] = max_value_to_ix
            max_value_to_ix += 1
        sampled_factor_ixs.append(value_to_ix[str(sampled_factor)])

    return sampled_factor_ixs, sampled_factors


def get_center_positions(digit_scales, num_cells=5):
    """
    Given digit scales, it returns (x,y) for top left portion of the image that ensures the object is centered
    :param digit_scales:
    :param num_cells:
    :return:
    """
    return [((num_cells - scale) // 2, (num_cells - scale) // 2) for scale in digit_scales]


def get_default_digit_position():
    return (0, 0)


def get_letters():
    # return ['a', 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'p']
    return ['a', 'c', 'k', 'm', 'n', 'p', 'r', 'u', 'v', 'w']


def get_letter_ord():
    return [ord(c) - ord('a') for c in get_letters()]


def get_letter_ord_to_ix():
    ords = get_letter_ord()
    letter_ord_to_ix = {}
    for ix, ord in enumerate(ords):
        letter_ord_to_ix[ord] = ix
    return letter_ord_to_ix


def load_letter_ix_to_images(split):
    """
    :param split:
    :return:
    """
    # Load the images and labels
    if split == 'test':
        images, labels = extract_test_samples('letters')
    else:
        images, labels = extract_training_samples('letters')

    # Create label to images
    letter_ix_to_images = {}
    valid_letter_ords = get_letter_ord()
    letter_ord_to_ix = get_letter_ord_to_ix()
    for ix, (img, l) in enumerate(zip(images, labels)):
        # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        label_ord = l - 1
        if label_ord in valid_letter_ords:
            letter_ix = letter_ord_to_ix[label_ord]
            if letter_ix not in letter_ix_to_images:
                letter_ix_to_images[letter_ix] = []
            letter_ix_to_images[letter_ix].append(img)

    # Perform train/val splits
    if split != 'test':
        for l in letter_ix_to_images:
            if split == 'val':
                letter_ix_to_images[l] = letter_ix_to_images[l][0:1000]
            else:
                letter_ix_to_images[l] = letter_ix_to_images[l][1000:]

    return letter_ix_to_images


def sample_conditional_biased_values(biased_value_list, biased_ixs, bias_ix_to_p_bias, avoid_ixs=None):
    """
    Samples biased value i.e., the value corresponding to biased_value_list[biased_ix] where bias_ix is the
    corresponding value specified in biased_ixs with probability p_bias

    If avoid_ixs is specified and if the biased_ix is same as avoid_ix, then it chooses something else at random

    :param biased_value_list: a map or an array from class ix to biased value based on some variable (size = num of classes)
    :param biased_ixs: Indicates the most prominent factor for each sample (size = num samples)
    :param bias_ix_to_p_bias: p_bias per bias value
    :param avoid_ixs: If biased_ix is same as avoid_ix then choose a different value at random
    :return:
    """
    if isinstance(biased_ixs, list):
        biased_ixs = np.asarray(biased_ixs)
    if avoid_ixs is not None and isinstance(avoid_ixs, list):
        avoid_ixs = np.asarray(avoid_ixs)

    all_factor_ixs = np.zeros_like(biased_ixs)
    all_factors = []
    unq_biased_ixs = np.unique(biased_ixs)

    for biased_ix in unq_biased_ixs:
        curr_ixs = np.where(biased_ixs == biased_ix)[0]
        curr_biased_ixs = biased_ixs[curr_ixs]
        curr_avoid_ixs = None
        if avoid_ixs is not None:
            curr_avoid_ixs = avoid_ixs[curr_ixs]
        sampled_factor_ixs, sampled_factors = sample_biased_values(biased_value_list, curr_biased_ixs,
                                                                   bias_ix_to_p_bias[biased_ix], curr_avoid_ixs)
        all_factor_ixs[curr_ixs] = sampled_factor_ixs

    for factor_ix in all_factor_ixs:
        all_factors.append(biased_value_list[factor_ix])

    return all_factor_ixs.tolist(), all_factors


def sample_biased_values(biased_value_list, biased_ixs, p_bias, avoid_ixs=None):
    """
    Samples biased value i.e., the value corresponding to biased_value_list[biased_ix] where bias_ix is the
    corresponding value specified in biased_ixs with probability p_bias

    If avoid_ixs is specified and if the biased_ix is same as avoid_ix, then it chooses something else at random

    :param biased_value_list: a map or an array from class ix to biased value based on some variable (size = num of classes)
    :param biased_ixs: Indicates the most prominent factor for each sample (size = num samples)
    :param p_bias:
    :param avoid_ixs: If biased_ix is same as avoid_ix then choose a different value at random
    :return:
    """
    if isinstance(biased_value_list, np.ndarray):
        biased_value_list = biased_value_list.tolist()
    value_to_ix = {}
    for cix, bv in enumerate(biased_value_list):
        value_to_ix[str(bv)] = cix

    class_ix_to_non_biased_values = get_non_biased_values_per_class(biased_value_list)

    # For a given sample, if bias_proba <= random_p, then the correlated/biased value for that factor is used
    # Else, we randomly sample from the remaining values
    random_p = np.random.random(len(biased_ixs))

    # Choose the values for each factor
    sampled_factor_ixs = []
    sampled_factors = []
    for ix, biased_ix in enumerate(biased_ixs):
        if random_p[ix] <= p_bias and (avoid_ixs is None or biased_ix != avoid_ixs[ix]):
            # Assign the biased value for each factor
            sampled_factor = biased_value_list[biased_ix]
        else:
            # Sample uniformly from rest of the factors
            rand_ix = int(np.random.choice(len(class_ix_to_non_biased_values[biased_ix]), size=1))
            sampled_factor = class_ix_to_non_biased_values[biased_ix][rand_ix]

        sampled_factors.append(sampled_factor)
        sampled_factor_ixs.append(value_to_ix[str(sampled_factor)])

    return sampled_factor_ixs, sampled_factors


def save_or_load_sampled_factors(save_dir, filename, rewrite=True, sampled_factors=None):
    """

    :param self:
    :param save_dir:
    :param rewrite:
    :param sampled_factors:
    :return:
    """
    factor_json_file = os.path.join(save_dir, f'{filename}.json')
    if not os.path.exists(factor_json_file) or rewrite:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, f'{filename}.json'), 'w') as f:
            json.dump(sampled_factors, f, indent=4, sort_keys=True)

    sampled_factors = json.load(open(factor_json_file))
    return sampled_factors


def count_groups(test):
    # file = '/hdd/robik/biased_mnist/full/test.json'
    # test = json.load(open(file))

    def get_grp_name(factor):
        return f'dig_{factor["digit"]}_col_{factor["digit_color_ix"]}_scale_{factor["digit_scale_ix"]}_' \
               f'texture_{factor["texture_ix"]}_txt_color_{factor["texture_color_ix"]}'

    factor_cnts = {}
    for f in test:
        grp_name = get_grp_name(f)
        if grp_name not in factor_cnts:
            factor_cnts[grp_name] = 0
        factor_cnts[grp_name] += 1

    print(json.dumps(factor_cnts, indent=4, sort_keys=True))
    print(len(factor_cnts))


def get_textures():
    return [' ', '-', '/', '|', '+', ')(', '~', '>', '::', '\\']


def get_natural_textures():
    return [
        'grass',
        'bark',
        'fabric',
        'plain',
        'wood',
        'brick',
        'wall',
        'waves',
        'noisy',
        'plain'
    ]


def get_default_texture():
    return ' '


def get_default_natural_texture():
    return 'plain'


def load_texture_images(textures_dir, scale=None):
    """
    Returns a map from filename to texture image
    :param textures_dir:
    :return:
    """
    textures = get_natural_textures()
    texture_images = {}

    for texture in textures:
        if texture not in texture_images:
            texture_images[texture] = []
        img_files = os.listdir(os.path.join(textures_dir, texture))
        for img_file in img_files:
            img = cv2.imread(os.path.join(textures_dir, texture, img_file))
            if scale is not None and scale != 1:
                img = cv2.resize(img, (img.shape[0] * scale, img.shape[1] * scale))
            texture_images[texture].append(img)
    return texture_images


def sample_texture_crops(num_samples, split, target_img_dim, texture_img_dim=1024, train_portion=2,
                         val_portion=1,
                         test_portion=1,
                         texture_img_scale=1):
    """
    Generate random numbers to select texture files and random crops within those texture files
    For ease, we assume that all full textures have dimensions of texture_img_dim x texture_img_dim

    :param texture_ixs:
    :param split:
    :param target_img_dim:
    :return:
    """
    # Select files at random
    # For each texture we choose a random value between 0 and 1 that will determine the actual image to be used
    texture_file_rand = np.random.random(num_samples)

    # For each image, we sample a random crop
    # For this we assume that the image has at least 1024 pixels in row/column.
    # We reserve first half (vertically) for train and 1/4th for val and remaining 1/4th for test
    column_width = texture_img_dim // (train_portion + val_portion + test_portion)
    if split == 'train':
        min_x1, max_x1 = 0, column_width * texture_img_scale - target_img_dim
    elif split == 'val':
        min_x1, max_x1 = column_width * texture_img_scale - target_img_dim, 2 * column_width * texture_img_scale - target_img_dim
    elif split == 'test':
        min_x1, max_x1 = 2 * column_width * texture_img_scale - target_img_dim, texture_img_dim * texture_img_scale - target_img_dim
    min_y1, max_y1 = 0, texture_img_dim * texture_img_scale - target_img_dim

    # Now create random crops
    x1s = np.random.randint(min_x1, max_x1, num_samples)
    x2s = x1s + target_img_dim
    y1s = np.random.randint(min_y1, max_y1, num_samples)
    y2s = y1s + target_img_dim
    return texture_file_rand, x1s, x2s, y1s, y2s


def to_list(p_bias):
    if isinstance(p_bias, list):
        return p_bias
    else:
        return [p_bias] * NUM_CLASSES


class NpEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def sample_long_tailed(class_ixs, class_imbalance_ratio, num_classes):
    """
    Returns item ixs for a new dataset with the specified imbalance ratio
    :param class_ixs:
    :param class_imbalance_ratio: (# instances in the rarest/# instance in the most frequent class)
    :param num_classes:
    :return:
    """
    if isinstance(class_ixs, list):
        class_ixs = np.asarray(class_ixs)
    num_data_per_class = get_num_data_per_class(class_ixs, class_imbalance_ratio, num_classes)
    sampled_item_ixs = []
    for class_ix in range(num_classes):
        all_item_ixs = np.where(class_ixs == class_ix)[0]
        np.random.shuffle(all_item_ixs)
        sampled_item_ixs += all_item_ixs[:num_data_per_class[class_ix]].tolist()
    return sampled_item_ixs


def get_num_data_per_class(class_ixs, class_imbalance_ratio, num_classes):
    """
    :param class_ixs: Class id for each sample
    :param class_imbalance_ratio: (# instances in the rarest/# instance in the most frequent class)
    :return: List of (item_ix, class_ix) pairs sampled at the given class imbalance ratio
    """
    max_data = len(class_ixs) / num_classes
    num_per_class = []
    for class_ix in range(num_classes):
        curr_num = max_data * (class_imbalance_ratio ** (class_ix / (num_classes - 1.0)))
        num_per_class.append(int(curr_num))
    return num_per_class
