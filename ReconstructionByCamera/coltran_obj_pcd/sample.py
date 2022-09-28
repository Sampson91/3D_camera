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

"""ColTran: Sampling scripts."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import numpy as np

import tensorflow.compat.v2 as tf

import datasets
from models import colorizer
from models import upsampler
from utils import base_utils
from utils import datasets_utils
from utils import train_utils

# pylint: disable=g-direct-tensorflow-import

# pylint: disable=missing-docstring
# pylint: disable=not-callable
# pylint: disable=g-long-lambda

flags.DEFINE_enum('mode', 'sample_test', [
    'sample_valid', 'sample_test', 'sample_train'], 'Operation mode.')

flags.DEFINE_string('logdir', '/tmp/svt', 'Main directory for logs.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU', 'TPU'],
                  'Hardware type.')

flags.DEFINE_enum('dataset', 'imagenet', ['imagenet', 'custom'], 'Dataset')
flags.DEFINE_string('data_dir', None, 'Data directory for custom images.')

flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_string('summaries_log_dir', 'summaries', 'Summaries parent.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('devices_per_worker', 1, 'Number of devices per worker.')
flags.DEFINE_integer('num_workers', 1, 'Number workers.')
config_flags.DEFINE_config_file(
    'config',
    default='test_configs/colorizer.py',
    help_string='Training configuration file.')

FLAGS = flags.FLAGS


def array_to_tf_example(array, label):
    """Converts array to a serialized TFExample."""
    array = np.ravel(array)
    array_list = tf.train.Int64List(value=array)
    label_list = tf.train.Int64List(value=np.array([label]))
    feature_dictionary = {
        'image': tf.train.Feature(int64_list=array_list),
        'label': tf.train.Feature(int64_list=label_list),
    }
    array_features = tf.train.Features(feature=feature_dictionary)
    example = tf.train.Example(features=array_features)
    return example.SerializeToString()


def build(config, batch_size, is_train=False):
    optimizer = train_utils.build_optimizer(config)
    exponetial_moving_average_variables = []

    downsample = config.get('downsample', False)
    downsample_resolution = config.get('downsample_resolution', 64)
    height, width = config.resolution

    if config.model.name == 'coltran_core':
        if downsample:
            height, width = downsample_resolution, downsample_resolution
        zero = tf.zeros((batch_size, height, width, 3), dtype=tf.int32)
        model = colorizer.ColTranCore(config.model)
        model(zero, training=is_train)

    current_channel = 1 if is_train else 3
    if config.model.name == 'color_upsampler':
        if downsample:
            height, width = downsample_resolution, downsample_resolution
        zero_slice = tf.zeros((batch_size, height, width, current_channel),
                              dtype=tf.int32)
        zero = tf.zeros((batch_size, height, width, 3), dtype=tf.int32)
        model = upsampler.ColorUpsampler(config.model)
        model(zero, inputs_slice=zero_slice, training=is_train)
    elif config.model.name == 'spatial_upsampler':
        zero_slice = tf.zeros((batch_size, height, width, current_channel),
                              dtype=tf.int32)
        zero = tf.zeros((batch_size, height, width, 3), dtype=tf.int32)
        model = upsampler.SpatialUpsampler(config.model)
        model(zero, inputs_slice=zero_slice, training=is_train)

    exponetial_moving_average_variables = model.trainable_variables
    exponetial_moving_average = train_utils.build_exponetial_moving_average(
        config, exponetial_moving_average_variables)
    return model, optimizer, exponetial_moving_average


def get_grayscale_at_sample_time(data, downsample_resolution, model_name):
    if model_name == 'spatial_upsampler':
        current_rgb = data['targets']
    else:
        current_rgb = data['targets_%d' % downsample_resolution]
    return tf.image.rgb_to_grayscale(current_rgb)


def create_sample_direction(logdir, config):
    """Creates child directory to write samples based on step name."""
    sample_direction = config.sample.get('log_dir')
    assert sample_direction is not None
    sample_direction = os.path.join(logdir, sample_direction)
    tf.io.gfile.makedirs(sample_direction)
    logging.info('writing samples at: %s', sample_direction)
    return sample_direction


def store_samples(data, config, logdir, generate_dataset=None):
    """Stores the generated samples."""
    downsample_resolution = config.get('downsample_resolution', 64)
    number_of_samples = config.sample.number_of_samples
    number_of_outputs = config.sample.number_of_outputs
    batch_size = config.sample.get('batch_size', 1)
    sample_mode = config.sample.get('mode', 'argmax')
    generate_file = config.sample.get('generate_file', 'gen')

    model, optimizer, exponetial_moving_average = build(config, 1, False)
    checkpoints = train_utils.create_checkpoint(model, optimizer,
                                                exponetial_moving_average)
    sample_direction = create_sample_direction(logdir, config)
    record_path = os.path.join(sample_direction, '%s.tfrecords' % generate_file)
    writer = tf.io.TFRecordWriter(record_path)

    train_utils.restore(model, checkpoints, logdir, exponetial_moving_average)
    number_of_training_steps = optimizer.iterations.numpy()
    logging.info('Producing sample after %d training steps.',
                 number_of_training_steps)

    logging.info(generate_dataset)
    for batch_indicator in range(number_of_outputs // batch_size):
        next_data = data.next()
        labels = next_data['label'].numpy()

        if generate_dataset is not None:
            next_generated_data = generate_dataset.next()

        # Gets grayscale image based on the model.
        current_gray = get_grayscale_at_sample_time(next_data,
                                                    downsample_resolution,
                                                    config.model.name)

        current_output = collections.defaultdict(list)
        for sample_indicator in range(number_of_samples):
            logging.info('Batch no: %d, Sample no: %d', batch_indicator,
                         sample_indicator)

            if config.model.name == 'color_upsampler':

                if generate_dataset is not None:
                    # Provide generated coarse color inputs.
                    scaled_rgb = next_generated_data['targets']
                else:
                    # Provide coarse color ground truth inputs.
                    scaled_rgb = next_data['targets_%d' % downsample_resolution]
                bit_rgb = base_utils.convert_bits(scaled_rgb,
                                                  number_of_bits_in=8,
                                                  number_of_bits_out=3)
                output = model.sample(gray_condition=current_gray,
                                      bit_condition=bit_rgb,
                                      mode=sample_mode)

            elif config.model.name == 'spatial_upsampler':
                if generate_dataset is not None:
                    # Provide low resolution generated image.
                    low_resolution = next_generated_data['targets']
                    low_resolution = datasets_utils.change_resolution(
                        low_resolution, 256)
                else:
                    # Provide low resolution ground truth image.
                    low_resolution = next_data[
                        'targets_%d_up_back' % downsample_resolution]
                output = model.sample(gray_condition=current_gray,
                                      inputs=low_resolution,
                                      mode=sample_mode)
            else:
                output = model.sample(gray_condition=current_gray,
                                      mode=sample_mode)
            logging.info('Done sampling')

            for out_key, out_item in output.items():
                current_output[out_key].append(out_item.numpy())

        # concatenate samples across width.
        for out_key, out_value in current_output.items():
            current_out_value = np.concatenate(out_value, axis=2)
            current_output[out_key] = current_out_value

            if ('sample' in out_key or 'argmax' in out_key):
                save_str = f'Saving {(batch_indicator + 1) * batch_size} samples'
                logging.info(save_str)
                for single_example, label in zip(current_out_value, labels):
                    serialized = array_to_tf_example(single_example, label)
                    writer.write(serialized)

    writer.close()


def sample(logdir, subset):
    """Executes the sampling loop."""
    logging.info('Beginning sampling loop...')
    config = FLAGS.config
    batch_size = config.sample.get('batch_size', 1)
    # used to parallelize sampling jobs.
    skip_batches = config.sample.get('skip_batches', 0)
    generate_data_direction = config.sample.get('generate_data_direction', None)
    is_generated = generate_data_direction is not None

    model_name = config.model.get('name')
    if not is_generated and 'upsampler' in model_name:
        logging.info('Generated low resolution not provided, using ground '
                     'truth input.')

    # Get ground truth dataset for grayscale image.
    tf_dataset = datasets.get_dataset(
        name=FLAGS.dataset,
        config=config,
        batch_size=batch_size,
        subset=subset,
        data_direction=FLAGS.data_dir)
    tf_dataset = tf_dataset.skip(skip_batches)
    data_iterator = iter(tf_dataset)

    # Creates dataset from generated TFRecords.
    # This is used as low resolution input to the upsamplers.
    generate_iterator = None
    if is_generated:
        generate_tf_dataset = datasets.get_generated_dataset(
            data_direction=generate_data_direction, batch_size=batch_size)
        generate_tf_dataset = generate_tf_dataset.skip(skip_batches)
        generate_iterator = iter(generate_tf_dataset)

    store_samples(data_iterator, config, logdir, generate_iterator)


def main(_):
    logging.info('Logging to %s.', FLAGS.logdir)
    if FLAGS.mode == 'sample_valid':
        logging.info('[main] I am the sampler.')
        sample(FLAGS.logdir, subset='valid')
    elif FLAGS.mode == 'sample_test':
        logging.info('[main] I am the sampler test.')
        sample(FLAGS.logdir, subset='test')
    elif FLAGS.mode == 'sample_train':
        logging.info('[main] I am the sampler train.')
        sample(FLAGS.logdir, subset='eval_train')
    else:
        raise ValueError(
            'Unknown mode {}. '
            'Must be one of [sample, sample_test]'.format(FLAGS.mode))


if __name__ == '__main__':
    app.run(main)
