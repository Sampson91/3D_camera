"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later
 used in subclasses.
"""
import random
from abc import ABC, abstractmethod

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, options).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, options):
        """Initialize the class; save the options in the class

        Parameters:
            options (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.options = options
        self.root = options.data_root

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or
                                test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(options, size):
    width, height = size
    new_height = height
    new_width = width
    if options.preprocess == 'resize_and_crop':
        new_height = new_width = options.load_size
    elif options.preprocess == 'scale_width_and_crop':
        new_width = options.load_size
        new_height = options.load_size * height // width

    width_value = random.randint(0, np.maximum(0, new_width - options.crop_size))
    height_value = random.randint(0, np.maximum(0, new_height - options.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (width_value, height_value), 'flip': flip}


def get_transform(options, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in options.preprocess:
        origin_size = [options.load_size, options.load_size]
        transform_list.append(transforms.Resize(origin_size, method))
    elif 'scale_width' in options.preprocess:
        transform_list.append(transforms.Lambda(lambda image: __scale_width(image, options.load_size,
                                                                            options.crop_size, method)))

    if 'crop' in options.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(options.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda image: __crop(image, params['crop_pos'], options.crop_size)))

    if options.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda image: __make_power_2(image, base=4, method=method)))

    if not options.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda image: __flip(image, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(image, base, method=transforms.InterpolationMode.BICUBIC):
    origin_width, origin_height = image.size
    height = int(round(origin_height / base) * base)
    width = int(round(origin_width / base) * base)
    if height == origin_height and width == origin_width:
        return image

    __print_size_warning(origin_width, origin_height, width, height)
    return image.resize((width, height), method)


def __scale_width(image, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    origin_width, origin_height = image.size
    if origin_width == target_size and origin_height >= crop_size:
        return image
    width = target_size
    height = int(max(target_size * origin_height / origin_width, crop_size))
    return image.resize((width, height), method)


def __crop(image, position, size):
    origin_width, origin_height = image.size
    position_x, position_y = position
    window_width = window_height = size
    if origin_width > window_width or origin_height > window_height:
        return image.crop((position_x, position_y, position_x + window_width, position_y + window_height))
    return image


def __flip(image, flip):
    if flip:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def __print_size_warning(origin_width, origin_height, width, height):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (origin_width, origin_height, width, height))
        __print_size_warning.has_printed = True
