#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
from . import networks
from .base_model import BaseModel


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or
                                test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, options):
        """Initialize the pix2pix class.

        Parameters:
            options (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert (not options.is_train)
        BaseModel.__init__(self, options)
        # specify the training losses you want to print out. The training/test scripts  will call
        # <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call
        # <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['generator' + options.model_suffix]  # only generator is needed.
        self.net_generator = networks.define_generator(options.input_num_channel, options.output_num_channel,
                                                       options.num_generator_filters, options.net_generator,
                                                       options.normalization, not options.no_dropout,
                                                       options.init_type, options.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'net_generator' + options.model_suffix, self.net_generator)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['origin'].to(self.device)
        self.image_paths = input['origin_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.net_generator(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
