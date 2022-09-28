#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix) and
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss
plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import print_current_losses

if __name__ == '__main__':
    options = TrainOptions().parse()    # get training options
    dataset = create_dataset(options)   # create a dataset given options.dataset_mode and other options
    dataset_size = len(dataset)         # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(options)       # create a model given opt.model and other options
    model.setup(options)                # regular setup: load and print networks; create schedulers
    total_iterations = 0                # the total number of training iterations

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(options.epoch_count, options.num_epochs + options.num_epochs_decay + 1):
        epoch_start_time = time.time()      # timer for entire epoch
        iteration_data_time = time.time()   # timer for data loading per iteration
        epoch_iteration = 0                 # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()        # update learning rates in the beginning of every epoch.
        for index, data in enumerate(dataset):  # inner loop within one epoch
            iteration_start_time = time.time()  # timer for computation per iteration
            if total_iterations % options.print_frequency == 0:
                time_data = iteration_start_time - iteration_data_time

            total_iterations += options.batch_size
            epoch_iteration += options.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.optimize_parameters()     # calculate loss functions, get gradients, update network weights

            # print training losses and save logging information to the disk
            if total_iterations % options.print_frequency == 0:
                losses = model.get_current_losses()
                time_comp = (time.time() - iteration_start_time) / options.batch_size
                print_current_losses(options, epoch, epoch_iteration, losses, time_comp, time_data)

            # cache our latest model every <save_latest_freq> iterations
            if total_iterations % options.save_latest_frequency == 0:
                print('saving the latest model (epoch %d, total_iterations %d)' % (epoch, total_iterations))
                save_suffix = 'iteration_%d' % total_iterations if options.save_by_iterate else 'latest'
                model.save_networks(save_suffix)

            iteration_data_time = time.time()
        if epoch % options.save_epoch_frequency == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iterations %d' % (epoch, total_iterations))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, options.num_epochs + options.num_epochs_decay, time.time() - epoch_start_time))
