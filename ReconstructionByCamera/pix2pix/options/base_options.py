#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
import argparse
import os

import torch

import data
import models
from util import util


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset
    class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.is_train = None
        self.parser = None
        self.options = None
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_root', required=True,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_directory', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='pix2pix', help='')
        parser.add_argument('--input_num_channel', type=int, default=1, help='# of input image channels: '
                                                                             '3 for RGB and 1 for grayscale')
        parser.add_argument('--output_num_channel', type=int, default=1, help='# of output image channels: '
                                                                              '3 for RGB and 1 for grayscale')
        parser.add_argument('--num_generator_filters', type=int, default=64, help='# of gen filters in'
                                                                                  ' the last conv layer')
        parser.add_argument('--num_discriminator_filters', type=int, default=64, help='# of discrimination filters'
                                                                                      'in the first conv layer')
        parser.add_argument('--net_discriminator', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model'
                                 ' is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--net_generator', type=str, default='unet_256',
                            help='specify generator architecture '
                                 '[resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--num_layers_discriminator', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normalization', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory'
                                 ' contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time '
                                 '[resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_windows_size', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iterate', type=int, default='0',
                            help='which iteration to load? if load_iterate > 0, the code will load models by '
                                 'iterate_[load_iterate]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: options.name = '
                                                                   'options.name + suffix: e.g., {model}_'
                                                                   '{networkGenerative}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        options, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = options.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)
        options, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = options.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.is_train)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, options):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for key, value in sorted(vars(options).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_directory = os.path.join(options.checkpoints_directory, options.name)
        util.mkdirs(expr_directory)
        file_name = os.path.join(expr_directory, '{}_options.txt'.format(options.phase))
        with open(file_name, 'wt') as option_file:
            option_file.write(message)
            option_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        options = self.gather_options()
        options.is_train = self.is_train  # train or test

        # process opt.suffix
        if options.suffix:
            suffix = ('_' + options.suffix.format(**vars(options))) if options.suffix != '' else ''
            options.name = options.name + suffix

        self.print_options(options)

        # set gpu ids
        gpu_str_ids = options.gpu_ids.split(',')
        options.gpu_ids = []
        for str_id in gpu_str_ids:
            id = int(str_id)
            if id >= 0:
                options.gpu_ids.append(id)
        if len(options.gpu_ids) > 0:
            torch.cuda.set_device(options.gpu_ids[0])

        self.options = options
        return self.options
