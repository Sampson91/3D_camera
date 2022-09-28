#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
import torch

from . import networks
from .base_model import BaseModel


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to
        output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
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

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, options):
        """Initialize the pix2pix class.

        Parameters:
            options (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, options)
        # specify the training losses you want to print out. The training/test scripts
        # will call <BaseModel.get_current_losses>
        self.loss_generator = None
        self.loss_generator_gan = None
        self.loss_generator_L1 = None
        self.fake_target = None
        self.loss_discriminator = None
        self.loss_discriminator_real = None
        self.loss_discriminator_fake = None
        self.real_target = None
        self.real_origin = None
        self.loss_names = ['generator_gan', 'generator_L1', 'discriminator_real', 'discriminator_fake']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        self.visual_names = ['real_origin', 'fake_target', 'real_target']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = ['generator', 'discriminator']
        else:  # during test time, only load Generator
            self.model_names = ['generator']
        # define networks (both generator and discriminator)
        self.net_generator = networks.define_generator(options.input_num_channel, options.output_num_channel,
                                                       options.num_generator_filters, options.net_generator,
                                                       options.norm, not options.no_dropout, options.init_type,
                                                       options.init_gain, self.gpu_ids)
        # define a discriminator; conditional GANs need to take both input and output images; Therefore,
        # channels for D is input_nc + output_nc
        if self.is_train:
            self.net_discriminator = networks.define_discriminator(options.input_num_channel +
                                                                   options.output_num_channel,
                                                                   options.num_discriminator_filters,
                                                                   options.net_discriminator,
                                                                   options.num_layers_discriminator,
                                                                   options.norm, options.init_type,
                                                                   options.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterion_gan = networks.GANLoss(options.gan_mode).to(self.device)
            self.criterion_L1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_generator = torch.optim.Adam(self.net_generator.parameters(), lr=options.learning_rate,
                                                        betas=(options.beta1, 0.999))
            self.optimizer_discriminator = torch.optim.Adam(self.net_discriminator.parameters(),
                                                            lr=options.learning_rate, betas=(options.beta1, 0.999))
            self.optimizers.append(self.optimizer_generator)
            self.optimizers.append(self.optimizer_discriminator)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        origin_to_target = self.options.direction == 'origin_to_target'
        self.real_origin = input['origin' if origin_to_target else 'target'].to(self.device)
        self.real_target = input['target' if origin_to_target else 'origin'].to(self.device)
        self.image_paths = input['origin_paths' if origin_to_target else 'target_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_target = self.net_generator(self.real_origin)  # Generator(origin)

    def backward_discriminator(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_target
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_origin_target = torch.cat((self.real_origin, self.fake_target), 1)
        pred_fake = self.net_discriminator(fake_origin_target.detach())
        self.loss_discriminator_fake = self.criterion_gan(pred_fake, False)
        # Real
        real_origin_target = torch.cat((self.real_origin, self.real_target), 1)
        pred_real = self.net_discriminator(real_origin_target)
        self.loss_discriminator_real = self.criterion_gan(pred_real, True)
        # combine loss and calculate gradients
        self.loss_discriminator = (self.loss_discriminator_fake + self.loss_discriminator_real) * 0.5
        self.loss_discriminator.backward()

    def backward_generator(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, Generator(origin) should fake the discriminator
        fake_origin_target = torch.cat((self.real_origin, self.fake_target), 1)
        pred_fake = self.net_discriminator(fake_origin_target)
        self.loss_generator_gan = self.criterion_gan(pred_fake, True)
        # Second, Generator(origin) = target
        self.loss_generator_L1 = self.criterion_L1(self.fake_target, self.real_target) * self.options.lambda_L1
        # combine loss and calculate gradients
        self.loss_generator = self.loss_generator_gan + self.loss_generator_L1
        self.loss_generator.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: Generator(origin)
        # update Discriminator
        self.set_requires_grad(self.net_discriminator, True)  # enable backprop for discriminator
        self.optimizer_discriminator.zero_grad()  # set Discriminator's gradients to zero
        self.backward_discriminator()  # calculate gradients for Discriminator
        self.optimizer_discriminator.step()  # update Discriminator's weights
        # update Generator
        self.set_requires_grad(self.net_discriminator, False)  # D requires no gradients when optimizing Generator
        self.optimizer_generator.zero_grad()  # set G's gradients to zero
        self.backward_generator()  # calculate graidents for Generator
        self.optimizer_generator.step()  # udpate Generator's weights
