#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/8 下午3:29
# @Author : wangyangyang
"""General-purpose test script for image-to-image translation.
Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util.util import save_images

if __name__ == '__main__':
    options = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    options.num_threads = 0     # test code only supports num_threads = 0
    options.batch_size = 1      # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    options.serial_batches = True
    options.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    dataset = create_dataset(options)   # create a dataset given opt.dataset_mode and other options
    model = create_model(options)       # create a model given opt.model and other options
    model.setup(options)                # regular setup: load and print networks; create schedulers

    save_path = os.path.join(options.results_directory, options.name,
                             '{}_{}'.format(options.phase, options.epoch))  # define the website directory
    print('creating save path', save_path)
    for index, data in enumerate(dataset):
        if index >= options.num_test:   # only apply our model to opt.num_test images.
            break
        model.set_input(data)           # unpack data from data loader
        model.test()                    # run inference
        visuals = model.get_current_visuals()   # get image results
        img_path = model.get_image_paths()      # get image paths
        if index % 5 == 0:
            print('processing (%04d)-th image... %s' % (index, img_path))
        save_images(save_path, visuals, img_path, aspect_ratio=options.aspect_ratio)
