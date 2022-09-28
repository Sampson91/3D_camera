#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()
        self.is_train = None

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_frequency', type=int, default=100, help='frequency of showing training '
                                                                             'results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_frequency', type=int, default=5000, help='frequency of saving'
                                                                                    ' the latest results')
        parser.add_argument('--save_epoch_frequency', type=int, default=5, help='frequency of saving checkpoints'
                                                                                ' at the end of epochs')
        parser.add_argument('--save_by_iterate', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model '
                                                                       'by <epoch_count>, <epoch_count>+'
                                                                       '<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs with the '
                                                                        'initial learning rate')
        parser.add_argument('--num_epochs_decay', type=int, default=100, help='number of epochs to linearly '
                                                                              'decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--learning_rate', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan |'
                                                                          ' wgangp]. vanilla GAN loss is the '
                                                                          'cross-entropy objective used in the '
                                                                          'original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores '
                                                                      'previously generated images')
        parser.add_argument('--learning_rate_policy', type=str, default='linear', help='learning rate policy. [linear |'
                                                                                       ' step | plateau | cosine]')
        parser.add_argument('--learning_rate_decay_iterations', type=int, default=50, help='multiply by a gamma every'
                                                                                           ' lr_decay_iters iterations')

        self.is_train = True
        return parser