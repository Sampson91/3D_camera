# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Script to colorize or recolorize a directory of images.

Instructions
------------
1. Download pretrained models from
https://storage.cloud.google.com/gresearch/coltran/coltran.zip

2. Set the following variables:

* LOGDIR    - Checkpoint Directory to the corresponding checkpoints.
* IMG_DIR   - Directory with ground-truth grayscale or colored images.
* STORE_DIR - Directory to store generated images.
* MODE      - "colorize" if IMG_DIR consists of grayscale images
              "recolorize" if IMG_DIR consists of colored images.

2. Run the colorizer to get a coarsely colorized image. Set as follows:

python -m coltran.custom_colorize --config=configs/colorizer.py \
--logdir=$LOGDIR/colorizer --img_dir=$IMG_DIR --store_dir=$STORE_DIR \
--mode=$MODE

The generated images will be stored in $STORE_DIR/stage1

Notes
-----
* The model is pre-trained on ImageNet. Colorized images may reflect the biases
present in the ImageNet dataset.
* Once in a while, there can be artifacts or anomalous colorizations
due to accumulation of errors.
See Section M of https://openreview.net/pdf?id=5NA1PinlGFu
* Legacy images may have a different distribution as compared to the
grayscale images used to train the model. This might reflect in difference in
colorization fidelity between colorizing legacy images and our reported results.
* Setting "mode" correctly is important.
If img_dir consists of grayscale images, it should be set to "colorize"
if img_dir consists of colored images , it should be set to "recolorize".

"""
import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
from ml_collections import config_flags
import numpy as np

import tensorflow.compat.v2 as tf

import datasets
from models import colorizer
from utils import base_utils
from utils import datasets_utils
from utils import train_utils

from prepare import main_output_index_and_image_once
from prepare import resize_image
from PIL import Image
from prepare import convert_spherically_out_image_and_reverse

from prepare import rgb_to_gray

flags.DEFINE_string('image_direction', None,
                    'Path for images needed to be colorized / recolorized.')
flags.DEFINE_string('logdir', './LOGDIR/',
                    'Main directory for logs.')
flags.DEFINE_string('generate_data_dir', None,
                    'Path to images generated from the previous stages. '
                    'Has to be set if the model is the color or spatial '
                    'upsampler.')
flags.DEFINE_string('store_direction', None, 'Path to store generated images.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU', 'TPU'],
                  'Hardware type.')
flags.DEFINE_enum('mode', 'colorize', ['colorize', 'recolorize'],
                  'Whether to colorizer or recolorize images.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('batch_size', None,
                     'Batch size. If not provided, use the optimal batch-size '
                     'for each model.')
# customized directions ↓
flags.DEFINE_string('upscaled_image_path', './prepare/upscaled_image',
                    'upscaled image directory')
flags.DEFINE_string('index_path', './prepare/test_index',
                    'Original index directory.')
flags.DEFINE_string('generated_image_direction', './prepare/store_image/stage1',
                    'Directory of images generated by colorizer.')

flags.DEFINE_integer('square_pixel_size', 256,
                     'Height and width of an image is the same.'
                     'Must be the same with the one in run.py')

flags.DEFINE_string('generated_image_index_path',
                    './prepare/generated_image_index',
                    'Index directory for generated image.')
flags.DEFINE_string('obj_saving_path', './prepare/generated_obj',
                    'Saving directory for generated obj.')

# flags.DEFINE_string('obj_direction', './prepare/obj',
#                     'Main directory for original obj files.')

flags.DEFINE_enum('add_face', 'False', ['False', 'True'],
                  'Add face back for evaluation. All information of the points, such as xyz, and sort, cannot be changed.')

flags.DEFINE_enum('force_convert_to_obj', 'No', ['No', 'Force'],
                  'Convert after stage 1.')

flags.DEFINE_string('image_rgb_direction', './prepare/test_image',
                    'Directory of saving rgb images.')

flags.DEFINE_string('image_gray_direction', './prepare/test_gray_image',
                    'Directory of saving gray images converted from rgb.')
# customized directions ↑

# flags for test_prepare() ↓
flags.DEFINE_string('test_obj_direction', './prepare/test_obj',
                    'Main directory for original obj files.')
flags.DEFINE_string('test_index_output_path', './prepare/test_index',
                    'Main output directory for original index.')
flags.DEFINE_string('test_image_saving_path', './prepare/test_image',
                    'Main directory for saving images.')
# flags for test_prepare() ↑


config_flags.DEFINE_config_file(
    'config',
    default='configs/colorizer.py',
    help_string='Training configuration file.')
FLAGS = flags.FLAGS


# Because some empty folders will not be uploaded to gitea,
# here write a function to check and create all required folders,
# which are not uploaded to gitea,
# in order to reduce errors and make it running smoother.
def check_required_folders():
    if not os.path.exists(FLAGS.store_direction):
        os.mkdir(FLAGS.store_direction)
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
    if not os.path.exists(FLAGS.generate_data_dir):
        os.mkdir(FLAGS.generate_data_dir)
    if not os.path.exists(FLAGS.upscaled_image_path):
        os.mkdir(FLAGS.upscaled_image_path)
    if not os.path.exists(FLAGS.generated_image_direction):
        os.mkdir(FLAGS.generated_image_direction)
    if not os.path.exists(FLAGS.obj_saving_path):
        os.mkdir(FLAGS.obj_saving_path)
    if not os.path.exists(FLAGS.image_gray_direction):
        os.mkdir(FLAGS.image_gray_direction)
    if not os.path.exists(FLAGS.image_rgb_direction):
        os.mkdir(FLAGS.image_rgb_direction)
    if not os.path.exists(FLAGS.generated_image_index_path):
        os.mkdir(FLAGS.generated_image_index_path)

    # flags for test_prepare() ↓
    if not os.path.exists(FLAGS.test_obj_direction):
        os.mkdir(FLAGS.test_obj_direction)
    if not os.path.exists(FLAGS.test_index_output_path):
        os.mkdir(FLAGS.test_index_output_path)
    if not os.path.exists(FLAGS.test_image_saving_path):
        os.mkdir(FLAGS.test_image_saving_path)


# flags for test_prepare() ↑

def test_prepare():
    obj_files = os.listdir(FLAGS.test_obj_direction)
    assert len(obj_files) > 0, 'no file in target direction'

    image_files = os.listdir(FLAGS.test_image_saving_path)
    assert len(obj_files) > 0 or len(image_files) > 0, (
        'no images and/or objs '
        'in target files')
    # if no image_files, convert all objs to images
    # otherwise, only convert new objs
    if len(image_files) != 0:

        print('checking and converting new objs to images')
        # check if there are new obj files need to be converted to image_files
        obj_names = []

        for obj_name in obj_files:
            obj_name_without_ext, _ = os.path.splitext(obj_name)
            i = 1
            for image_name in image_files:
                image_name_without_ext, _ = os.path.splitext(image_name)
                if obj_name_without_ext == image_name_without_ext:
                    i = 0
                    break
            if i:
                obj_names.append(obj_name)

        if len(obj_names) == 0:
            print('no new objs added and no images are converted from new objs')
        else:
            print('the list of objs will be converted to images', obj_names)

    else:
        print('convert all objs to images')
        obj_names = obj_files

    for obj_file in obj_names:
        main_output_index_and_image_once.from_obj_to_index_and_image(
            obj_direction=FLAGS.test_obj_direction, obj_file=obj_file,
            index_output_path=FLAGS.test_index_output_path,
            image_saving_path=FLAGS.test_image_saving_path,
            square_pixel_size=FLAGS.square_pixel_size)


def convert_argb_to_rgb():
    path = FLAGS.generated_image_direction
    file_list = os.listdir(path)
    for file in file_list:
        all_path = os.path.join(path, file)
        resize_image.argb_convert_to_rgb(path_with_file=all_path)


def rgb_to_gray_main():
    rgb_image_path = FLAGS.image_rgb_direction
    gray_image_output = FLAGS.image_gray_direction

    rgb_file_list = os.listdir(rgb_image_path)
    gray_file_list = os.listdir(gray_image_output)

    if not len(gray_file_list):
        i = 1
        for rgb_file_ in rgb_file_list:
            i = 1
            for gray_file_ in gray_file_list:
                if gray_file_ == rgb_file_:
                    i = 0
                    break
            if i:
                rgb_to_gray.rgb_to_gray_function(path=rgb_image_path,
                                                 file=rgb_file_,
                                                 out_path=gray_image_output)
        print('convert from rgb to gray complete')
    else:
        print('no rgb image needs to be converted to gray')


def resize_image_main():
    print('start resizing')
    destination_path = FLAGS.upscaled_image_path
    files_need_to_copy = os.listdir(FLAGS.generated_image_direction)
    assert len(files_need_to_copy) > 0, (
        'no image has been generated by custom_colorize')

    scale = FLAGS.square_pixel_size

    for file_ in files_need_to_copy:
        copy_file_with_direction = os.path.join(FLAGS.generated_image_direction,
                                                file_)
        resize_image.copy_file(copy_file_with_direction, destination_path)
        resize_path = os.path.join(destination_path, file_)
        resize_image.resize_image_function(image_input=resize_path,
                                           image_output=resize_path,
                                           pixel=scale)
    print('resize complete')


def convert_back_to_obj():
    print('start converting')
    resized_image_direction = FLAGS.upscaled_image_path

    index_files = os.listdir(FLAGS.index_path)
    assert len(index_files) > 0, 'no file in index direction'
    resized_image_files = os.listdir(resized_image_direction)
    assert len(
        resized_image_files) > 0, 'no file in generated image direction'

    for index_file in index_files:
        index_name_without_ext, _ = os.path.splitext(index_file)
        for image_name in resized_image_files:
            image_name_without_ext, _ = os.path.splitext(image_name)
            if index_name_without_ext == image_name_without_ext:
                main_output_index_and_image_once.convert_generated_image_to_obj_reversely(
                    resized_image_direction=resized_image_direction,
                    generated_image_file_by_splitting_from_index=image_name,
                    index_path=FLAGS.index_path,
                    index_file=index_file,
                    generated_image_index_path=FLAGS.generated_image_index_path,
                    obj_saving_path=FLAGS.obj_saving_path)
    print('convert from resized_image to obj complete')


def add_original_face_information_to_generated_obj():
    save_path = './prepare/obj_with_face'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    original_obj_files = os.listdir(FLAGS.test_obj_direction)
    reversed_obj_files = os.listdir(FLAGS.obj_saving_path)
    added_list = []
    not_added_list = original_obj_files
    for original_obj_file_ in original_obj_files:
        for reversed_obj_file_ in reversed_obj_files:
            if original_obj_file_ == reversed_obj_file_:
                did_or_not = convert_spherically_out_image_and_reverse.add_face_to_reversed_obj(
                    original_obj_path=FLAGS.test_obj_direction,
                    original_obj_file=original_obj_file_,
                    reversed_obj_path=FLAGS.obj_saving_path,
                    reversed_obj_file=reversed_obj_file_,
                    save_path=save_path)

                if did_or_not:
                    added_list.append(reversed_obj_file_)
                    not_added_list.remove(reversed_obj_file_)

    if len(added_list):
        print('faces added to the following list of objs', added_list)
    else:
        print('no faces added to reversed obj')
    if len(not_added_list):
        print(
            'faces in the following list of objs are not added to reversed objs:',
            not_added_list)
    else:
        print('all faces from original objs are added to reversed objs')


def create_grayscale_dataset_from_images(image_direction, batch_size):
    """Creates a dataset of grayscale images from the input image directory."""

    def load_and_preprocess_image(path, child_path):
        image_str = tf.io.read_file(path)
        number_of_channels = 1 if FLAGS.mode == 'colorize' else 3
        image = tf.image.decode_image(image_str, channels=number_of_channels)

        # Central crop to square and resize to 256x256.
        image = datasets.resize_to_square(image, resolution=256,
                                          train=False)

        # Resize to a low resolution image.
        image_64 = datasets_utils.change_resolution(image, resolution=64)
        if FLAGS.mode == 'recolorize':
            image = tf.image.rgb_to_grayscale(image)
            image_64 = tf.image.rgb_to_grayscale(image_64)
        return image, image_64, child_path

    child_files = tf.io.gfile.listdir(image_direction)
    files = [os.path.join(image_direction, file) for file in child_files]
    files = tf.convert_to_tensor(files, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((files, child_files))
    dataset = dataset.map(load_and_preprocess_image)
    return dataset.batch(batch_size=batch_size)


def build_model(config):
    """Builds model."""
    name = config.model.name
    optimizer = train_utils.build_optimizer(config)

    zero_64 = tf.zeros((1, 64, 64, 3), dtype=tf.int32)
    zero_64_slice = tf.zeros((1, 64, 64, 1), dtype=tf.int32)
    zero = tf.zeros((1, 256, 256, 3), dtype=tf.int32)
    zero_slice = tf.zeros((1, 256, 256, 1), dtype=tf.int32)

    if name == 'coltran_core':
        model = colorizer.ColTranCore(config.model)
        model(zero_64, training=False)
    # elif name == 'color_upsampler':
    #     model = upsampler.ColorUpsampler(config.model)
    #     model(inputs=zero_64, inputs_slice=zero_64_slice, training=False)
    # elif name == 'spatial_upsampler':
    #     model = upsampler.SpatialUpsampler(config.model)
    #     model(inputs=zero, inputs_slice=zero_slice, training=False)

    exponetial_moving_average_variables = model.trainable_variables
    exponetial_moving_average = train_utils.build_exponetial_moving_average(
        config, exponetial_moving_average_variables)
    return model, optimizer, exponetial_moving_average


def get_batch_size(name):
    """Gets optimal batch-size based on model."""
    if FLAGS.batch_size is not None:
        return FLAGS.batch_size
    elif 'upsampler' in name:
        return 5
    return 1


def get_store_direction(name, store_direction):
    store_dictionary = {
        'coltran_core': 'stage1',
        'color_upsampler': 'stage2',
        'spatial_upsampler': 'final'
    }
    store_direction = os.path.join(store_direction, store_dictionary[name])
    tf.io.gfile.makedirs(store_direction)
    return store_direction


def main(_):
    # check missing folders and create
    check_required_folders()

    # prepare for testing
    test_prepare()

    # convert rgb image to gray image
    rgb_to_gray_main()

    config, store_direction, image_direction = FLAGS.config, FLAGS.store_direction, FLAGS.image_direction
    assert store_direction is not None
    assert image_direction is not None
    model_name, generated_data_direction = config.model.name, FLAGS.generate_data_dir
    needs_generation = model_name in ['color_upsampler', 'spatial_upsampler']

    batch_size = get_batch_size(model_name)
    store_direction = get_store_direction(model_name, store_direction)
    number_of_files = len(tf.io.gfile.listdir(image_direction))

    if needs_generation:
        assert generated_data_direction is not None
        generated_dataset = datasets.create_generated_dataset_from_images(
            generated_data_direction)
        generated_dataset = generated_dataset.batch(batch_size)
        gen_dataset_iter = iter(generated_dataset)

    dataset = create_grayscale_dataset_from_images(FLAGS.image_direction,
                                                   batch_size)
    dataset_iter = iter(dataset)

    model, optimizer, exponetial_moving_average = build_model(config)
    checkpoints = train_utils.create_checkpoint(model, optimizer=optimizer,
                                                exponetial_moving_average=exponetial_moving_average)
    train_utils.restore(model, checkpoints, FLAGS.logdir,
                        exponetial_moving_average)
    number_of_training_steps = optimizer.iterations.numpy()
    logging.info('Producing sample after %d training steps.',
                 number_of_training_steps)

    number_of_epochs = int(np.ceil(number_of_files / batch_size))
    logging.info(number_of_epochs)

    for _ in range(number_of_epochs):
        gray, gray_64, child_paths = next(dataset_iter)

        if needs_generation:
            previous_generated = next(gen_dataset_iter)

        if model_name == 'coltran_core':
            out = model.sample(gray_64, mode='sample')
            samples = out['auto_sample']
        elif model_name == 'color_upsampler':
            previous_generated = base_utils.convert_bits(previous_generated,
                                                         number_of_bits_in=8,
                                                         number_of_bits_out=3)
            out = model.sample(bit_condition=previous_generated,
                               gray_condition=gray_64)
            samples = out['bit_up_argmax']
        else:
            previous_generated = datasets_utils.change_resolution(
                previous_generated, 256)
            out = model.sample(gray_condition=gray, inputs=previous_generated,
                               mode='argmax')
            samples = out['high_res_argmax']

        child_paths = child_paths.numpy()
        child_paths = [child_path.decode('utf-8') for child_path in child_paths]
        logging.info(child_paths)

        for sample, child_path in zip(samples, child_paths):
            write_path = os.path.join(store_direction, child_path)
            logging.info(write_path)
            sample = sample.numpy().astype(np.uint8)

            logging.info(sample.shape)
            with tf.io.gfile.GFile(write_path, 'wb') as f:
                plt.imsave(f, sample)

    '''
    convert output 64*64 image to needed image size
    than convert to obj
    '''
    # convert the output from argb to rgb
    convert_argb_to_rgb()

    # resize 64*64 image to 512*512
    resize_image_main()

    # convert resized image to obj
    convert_back_to_obj()

    # default FLAGS.add_face == False
    # if processed obj need to be checked visually
    # set (--add_face=True) in parameter
    if FLAGS.add_face == 'True':
        add_original_face_information_to_generated_obj()


if __name__ == '__main__':
    app.run(main)