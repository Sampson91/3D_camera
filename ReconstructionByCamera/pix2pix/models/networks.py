#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, value):
        return value


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(value):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, options):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        options (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if options.learning_rate_policy == 'linear':
        def lambda_rule(epoch):
            learning_rate_linear = 1.0 - max(0, epoch + options.epoch_count -
                                             options.num_epochs) / float(options.num_epochs_decay + 1)
            return learning_rate_linear

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif options.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=options.lr_decay_iters, gamma=0.1)
    elif options.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif options.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.num_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', options.learing_rate_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(model):  # define the initialization function
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(model.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(model.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(model.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(model, 'bias') and model.bias is not None:
                init.constant_(model.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(model.weight.data, 1.0, init_gain)
            init.constant_(model.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_generator(input_num_channel, output_num_channel, num_generator_filters, net_generator, norm='batch',
                     use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_num_channel (int) -- the number of channels in input images
        output_num_channel (int) -- the number of channels in output images
        num_generator_filters (int) -- the number of filters in the last conv layer
        net_generator (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project
         (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if net_generator == 'resnet_9blocks':
        net = ResnetGenerator(input_num_channel, output_num_channel, num_generator_filters, norm_layer=norm_layer,
                              use_dropout=use_dropout, num_blocks=9)
    elif net_generator == 'resnet_6blocks':
        net = ResnetGenerator(input_num_channel, output_num_channel, num_generator_filters, norm_layer=norm_layer,
                              use_dropout=use_dropout, num_blocks=6)
    elif net_generator == 'unet_128':
        net = UnetGenerator(input_num_channel, output_num_channel, 7, num_generator_filters, norm_layer=norm_layer,
                            use_dropout=use_dropout)
    elif net_generator == 'unet_256':
        net = UnetGenerator(input_num_channel, output_num_channel, 8, num_generator_filters, norm_layer=norm_layer,
                            use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_generator)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_discriminator(input_num_channel, num_discriminator_filters, net_discriminator, num_layers_discriminator=3,
                         norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_num_channel (int)     -- the number of channels in input images
        num_discriminator_filters (int)          -- the number of filters in the first conv layer
        net_discriminator (str)         -- the architecture's name: basic | n_layers | pixel
        num_layers_discriminator (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if net_discriminator == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_num_channel, num_discriminator_filters, num_layers=3, norm_layer=norm_layer)
    elif net_discriminator == 'num_layers':  # more options
        net = NLayerDiscriminator(input_num_channel, num_discriminator_filters,
                                  num_layers_discriminator, norm_layer=norm_layer)
    elif net_discriminator == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_num_channel, num_discriminator_filters, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net_discriminator)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_num_channel, output_num_channel, num_generator_filters=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, num_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_num_channel (int)      -- the number of channels in input images
            output_num_channel (int)     -- the number of channels in output images
            num_generator_filters (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            num_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (num_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_num_channel, num_generator_filters, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(num_generator_filters),
                 nn.ReLU(True)]

        num_down_sampling = 2
        for index in range(num_down_sampling):  # add downsampling layers
            mult = 2 ** index
            model += [nn.Conv2d(num_generator_filters * mult, num_generator_filters * mult * 2, kernel_size=3, stride=2,
                                padding=1, bias=use_bias),
                      norm_layer(num_generator_filters * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** num_down_sampling
        for index in range(num_blocks):  # add ResNet blocks

            model += [ResnetBlock(num_generator_filters * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for index in range(num_down_sampling):  # add up sampling layers
            mult = 2 ** (num_down_sampling - index)
            model += [nn.ConvTranspose2d(num_generator_filters * mult, int(num_generator_filters * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(num_generator_filters * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(num_generator_filters, output_num_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dimension, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dimension, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dimension, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dimension (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        padding = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dimension, dimension, kernel_size=3, padding=padding, bias=use_bias),
                       norm_layer(dimension), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        padding = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dimension, dimension, kernel_size=3, padding=padding, bias=use_bias),
                       norm_layer(dimension)]

        return nn.Sequential(*conv_block)

    def forward(self, value):
        """Forward function (with skip connections)"""
        out = value + self.conv_block(value)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_num_channel, output_num_channel, num_downs, num_generator_filters=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_num_channel (int)  -- the number of channels in input images
            output_num_channel (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            num_generator_filters (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(num_generator_filters * 8, num_generator_filters * 8,
                                             input_num_channel=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for index in range(num_downs - 5):  # add intermediate layers with num generator filters * 8 filters
            unet_block = UnetSkipConnectionBlock(num_generator_filters * 8, num_generator_filters * 8,
                                                 input_num_channel=None, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        # gradually reduce the number of filters from num generator filters * 8 to num generator filters
        unet_block = UnetSkipConnectionBlock(num_generator_filters * 4, num_generator_filters * 8,
                                             input_num_channel=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_generator_filters * 2, num_generator_filters * 4,
                                             input_num_channel=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_generator_filters, num_generator_filters * 2, input_num_channel=None,
                                             submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_num_channel, num_generator_filters,
                                             input_num_channel=input_num_channel, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, value):
        """Standard forward"""
        return self.model(value)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_num_channel, inner_num_channel, input_num_channel=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_num_channel (int) -- the number of filters in the outer conv layer
            inner_num_channel (int) -- the number of filters in the inner conv layer
            input_num_channel (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_num_channel is None:
            input_num_channel = outer_num_channel
        down_conv = nn.Conv2d(input_num_channel, inner_num_channel, kernel_size=4,
                              stride=2, padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_num_channel)
        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_num_channel)

        if outermost:
            up_conv = nn.ConvTranspose2d(inner_num_channel * 2, outer_num_channel,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_num_channel, outer_num_channel,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.ConvTranspose2d(inner_num_channel * 2, outer_num_channel,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, value):
        if self.outermost:
            return self.model(value)
        else:  # add skip connections
            return torch.cat([value, self.model(value)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_num_channel, num_discriminator_filters=64, num_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_num_channel (int)  -- the number of channels in input images
            num_discriminator_filters (int)       -- the number of filters in the last conv layer
            num_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence = [
            nn.Conv2d(input_num_channel, num_discriminator_filters, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)]
        num_filters_mult = 1
        nf_mult_prev = 1
        for num in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = num_filters_mult
            num_filters_mult = min(2 ** num, 8)
            sequence += [
                nn.Conv2d(num_discriminator_filters * nf_mult_prev, num_discriminator_filters * num_filters_mult,
                          kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(num_discriminator_filters * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = num_filters_mult
        num_filters_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(num_discriminator_filters * nf_mult_prev, num_discriminator_filters * num_filters_mult,
                      kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(num_discriminator_filters * num_filters_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(num_discriminator_filters * num_filters_mult, 1, kernel_size=kernel_size, stride=1,
                               padding=padding)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, value):
        """Standard forward."""
        return self.model(value)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_num_channel, num_discriminator_filters=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_num_channel (int)  -- the number of channels in input images
            num_discriminator_filters (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_num_channel, num_discriminator_filters, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_discriminator_filters, num_discriminator_filters * 2, kernel_size=1, stride=1, padding=0,
                      bias=use_bias),
            norm_layer(num_discriminator_filters * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_discriminator_filters * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, value):
        """Standard forward."""
        return self.net(value)
