"""This module contains simple helper functions """
from __future__ import print_function

import ntpath
import os

import numpy as np
import torch
from PIL import Image


def tensor_to_image_array(input_image, image_type=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        image_type (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(image_type)


def diagnose_network(network, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        network (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in network.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pillow = Image.fromarray(image_numpy)
    height, width, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pillow = image_pillow.resize((height, int(width * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pillow = image_pillow.resize((int(height / aspect_ratio), width), Image.BICUBIC)
    image_pillow.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_current_losses(options, epoch, iterations, losses, time_comp, time_data):
    """print current losses on console; also save the losses to the disk

    Parameters:
        options (TrainOptions) -- model options
        epoch (int) -- current epoch
        iterations (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        time_comp (float) -- computational time per data point (normalized by batch_size)
        time_data (float) -- data loading time per data point (normalized by batch_size)
    """
    log_name = os.path.join(options.checkpoints_directory, options.name, 'loss_log.txt')
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iterations, time_comp, time_data)
    for key, value in losses.items():
        message += '%s: %.3f ' % (key, value)

    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def save_images(image_directory, visuals, image_path, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        image_directory (str)          -- save image dir
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, image_data in visuals.items():
        image = tensor_to_image_array(image_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_directory, image_name)
        save_image(image, save_path, aspect_ratio=aspect_ratio)
