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

"""ColTran: Training and Continuous Evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
import tensorflow as tf
import tensorflow_datasets as tensorflow_datasets

import datasets
from models import colorizer

from utils import train_utils

from prepare import main_output_index_and_image_once

# pylint: disable=g-direct-tensorflow-import

# pylint: disable=missing-docstring
# pylint: disable=not-callable
# pylint: disable=g-long-lambda

flags.DEFINE_enum('mode', 'train', [
    'train', 'eval_train', 'eval_valid', 'eval_test'], 'Operation mode.')

flags.DEFINE_string('logdir', './LOGDIR/',
                    'Main directory for logs.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU', 'TPU'],
                  'Hardware type.')
flags.DEFINE_enum('dataset', 'custom', ['imagenet', 'custom'], 'Dataset')
flags.DEFINE_string('data_dir', None, 'Data directory for custom images.')
flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_string(
    'pretrain_dir', None, 'Finetune from a pretrained checkpoint.')
flags.DEFINE_string('summaries_log_dir', 'summaries', 'Summaries parent.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('devices_per_worker', 1, 'Number of devices per worker.')
flags.DEFINE_integer('num_workers', 1, 'Number workers.')

# flags for prepare() ↓
flags.DEFINE_string('obj_direction', './prepare/obj',
                    'Main directory for original obj files.')
flags.DEFINE_string('index_output_path', './prepare/index',
                    'Main output directory for original index.')
flags.DEFINE_string('image_saving_path', './prepare/image',
                    'Main directory for saving images.')
flags.DEFINE_integer('square_pixel_size', 256,
                     'Height and width of an image is the same.')

flags.DEFINE_enum('convert_obj_to_image', 'False', ['False', 'True'],
                  'Select if need to convert to image')

# flags for prepare() ↑

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
    if not os.path.exists(FLAGS.index_output_path):
        os.mkdir(FLAGS.index_output_path)
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
    if not os.path.exists(FLAGS.image_saving_path):
        os.mkdir(FLAGS.image_saving_path)


def prepare():
    obj_files = os.listdir(FLAGS.obj_direction)
    assert len(obj_files) > 0, 'no file in target direction'

    image_files = os.listdir(FLAGS.image_saving_path)
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
            obj_direction=FLAGS.obj_direction, obj_file=obj_file,
            index_output_path=FLAGS.index_output_path,
            image_saving_path=FLAGS.image_saving_path,
            square_pixel_size=FLAGS.square_pixel_size)


def restore_checkpoint(model, exponetial_moving_average, strategy,
                       latest_check_point=None, optimizer=None):
    if optimizer is None:
        check_point_function = functools.partial(
            train_utils.create_checkpoint, models=model,
            exponetial_moving_average=exponetial_moving_average)
    else:
        check_point_function = functools.partial(
            train_utils.create_checkpoint, models=model,
            exponetial_moving_average=exponetial_moving_average,
            optimizer=optimizer)

    checkpoint = train_utils.with_strategy(check_point_function, strategy)
    if latest_check_point:
        logging.info('Restoring from pretrained directory: %s',
                     latest_check_point)
        train_utils.with_strategy(
            lambda: checkpoint.restore(latest_check_point),
            strategy)
    return checkpoint


def is_tpu():
    return FLAGS.accelerator_type == 'TPU'


def loss_on_batch(inputs, model, config, training=False):
    """Loss on a batch of inputs."""
    logits, auxiliary_output = model.get_logits(
        inputs_dictionary=inputs, train_config=config, training=training)
    loss, auxiliary_loss_dict = model.loss(
        targets=inputs, logits=logits, train_config=config, training=training,
        auxiliary_output=auxiliary_output)  # tf_loss 使用aux_output
    loss_factor = config.get('loss_factor', 1.0)

    loss_dictionary = collections.OrderedDict()
    loss_dictionary['loss'] = loss
    total_loss = loss_factor * loss

    for auxiliary_key, auxiliary_loss in auxiliary_loss_dict.items():
        auxiliary_loss_factor = config.get(f'{auxiliary_key}_loss_factor', 1.0)
        loss_dictionary[auxiliary_key] = auxiliary_loss
        total_loss += auxiliary_loss_factor * auxiliary_loss
    loss_dictionary['total_loss'] = total_loss

    extra_info = collections.OrderedDict([
        ('scalar', loss_dictionary),
    ])
    return total_loss, extra_info


def train_step(config,
               model,
               optimizer,
               metrics,
               exponetial_moving_average=None,
               strategy=None):
    """Training StepFn."""

    def step_function(inputs):
        """Per-Replica StepFn."""
        with tf.GradientTape() as tape:
            loss, extra = loss_on_batch(inputs, model, config, training=True)
            scaled_loss = loss
            if strategy:
                scaled_loss /= float(strategy.num_replicas_in_sync)
                # strategy 在train_utils->setup_strategy
                # strategy = tf.distribute.experimental.TPUStrategy(cluster)
                # 库函数中调用num_replicas_in_sync

        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for metric_key, metric in metrics.items():
            metric.update_state(extra['scalar'][metric_key])

        if exponetial_moving_average is not None:
            exponetial_moving_average.apply(model.trainable_variables)
        return loss

    return train_utils.step_with_strategy(step_function, strategy)


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

    # current_channel = 1 if is_train else 3
    # if config.model.name == 'color_upsampler':
    #     if downsample:
    #         height, width = downsample_resolution, downsample_resolution
    #     zero_slice = tf.zeros((batch_size, height, width, current_channel),
    #                           dtype=tf.int32)
    #     zero = tf.zeros((batch_size, height, width, 3), dtype=tf.int32)
    #     model = upsampler.ColorUpsampler(config.model)
    #     model(zero, inputs_slice=zero_slice, training=is_train)
    # elif config.model.name == 'spatial_upsampler':
    #     zero_slice = tf.zeros((batch_size, height, width, current_channel),
    #                           dtype=tf.int32)
    #     zero = tf.zeros((batch_size, height, width, 3), dtype=tf.int32)
    #     model = upsampler.SpatialUpsampler(config.model)
    #     model(zero, inputs_slice=zero_slice, training=is_train)

    exponetial_moving_average_variables = model.trainable_variables
    exponetial_moving_average = train_utils.build_exponetial_moving_average(
        config, exponetial_moving_average_variables)
    return model, optimizer, exponetial_moving_average


###############################################################################
## Train.
###############################################################################
def train(logdir):
    config = FLAGS.config
    steps_per_write = FLAGS.steps_per_summaries
    train_utils.write_config(config, logdir)

    strategy, batch_size = train_utils.setup_strategy(
        config, FLAGS.master,
        FLAGS.devices_per_worker, FLAGS.mode, FLAGS.accelerator_type)

    def input_function(input_context=None):
        read_config = None
        if input_context is not None:
            read_config = tensorflow_datasets.ReadConfig(
                input_context=input_context)

        dataset = datasets.get_dataset(
            name=FLAGS.dataset,
            config=config,
            batch_size=config.batch_size,
            subset='train',
            read_config=read_config,
            data_direction=FLAGS.data_dir)
        return dataset

    # DATASET CREATION.
    logging.info('Building dataset.')
    train_dataset = train_utils.dataset_with_strategy(input_function, strategy)
    data_iterator = iter(train_dataset)

    # MODEL BUILDING
    logging.info('Building model.')
    model, optimizer, exponetial_moving_average = train_utils.with_strategy(
        lambda: build(config, batch_size, True), strategy)
    model.summary(120, print_fn=logging.info)

    # METRIC CREATION.
    metrics = {}
    metric_keys = ['loss', 'total_loss']
    metric_keys += model.metric_keys
    for metric_key in metric_keys:
        funciton = functools.partial(tf.keras.metrics.Mean, metric_key)
        current_metric = train_utils.with_strategy(funciton, strategy)
        metrics[metric_key] = current_metric

    # CHECKPOINTING LOGIC.
    if FLAGS.pretrain_dir is not None:
        pretrain_check_point = tf.train.latest_checkpoint(FLAGS.pretrain_dir)
        assert pretrain_check_point

        # Load the entire model without the optimizer from the checkpoints.
        restore_checkpoint(model, exponetial_moving_average, strategy,
                           pretrain_check_point, optimizer=None)
        # New tf.train.Checkpoint instance with a reset optimizer.
        checkpoint = restore_checkpoint(
            model, exponetial_moving_average, strategy, latest_check_point=None,
            optimizer=optimizer)
    else:
        latest_check_point = tf.train.latest_checkpoint(logdir)
        checkpoint = restore_checkpoint(
            model, exponetial_moving_average, strategy, latest_check_point,
            optimizer=optimizer)

    checkpoint = tf.train.CheckpointManager(
        checkpoint, directory=logdir, checkpoint_name='model', max_to_keep=10)
    if optimizer.iterations.numpy() == 0:
        checkpoint_name = checkpoint.save()
        logging.info('Saved checkpoint to %s', checkpoint_name)

    train_summary_directory = os.path.join(logdir, 'train_summaries')
    writer = tf.summary.create_file_writer(train_summary_directory)
    start_time = time.time()

    logging.info('Start Training.')

    # This hack of wrapping up multiple train steps with a tf.function call
    # speeds up training significantly.
    # See: https://www.tensorflow.org/guide/tpu#improving_performance_by_multiple_steps_within_tffunction # pylint: disable=line-too-long
    @tf.function
    def train_multiple_steps(iterator, steps_per_epoch):

        train_step_function = train_step(config, model, optimizer, metrics,
                                         exponetial_moving_average,
                                         strategy)

        for _ in range(steps_per_epoch):
            train_step_function(iterator)

    while optimizer.iterations.numpy() < config.get('max_train_steps', 1000000):
        number_of_train_steps = optimizer.iterations

        for metric_key in metric_keys:
            metrics[metric_key].reset_states()

        start_run = time.time()

        train_multiple_steps(data_iterator,
                             tf.convert_to_tensor(steps_per_write))

        steps_per_sec = steps_per_write / (time.time() - start_run)
        with writer.as_default():
            for metric_key, metric in metrics.items():
                metric_numpy = metric.result().numpy()
                tf.summary.scalar(metric_key, metric_numpy,
                                  step=number_of_train_steps)

                if metric_key == 'total_loss':
                    logging.info(
                        'Loss: %.3f bits/dimension, Speed: %.3f steps/second',
                        metric_numpy, steps_per_sec)
        if time.time() - start_time > config.save_checkpoint_secs:
            checkpoint_name = checkpoint.save()
            logging.info('Saved checkpoint to %s', checkpoint_name)
            start_time = time.time()


###############################################################################
## Evaluating.
###############################################################################


def evaluate(logdir, subset):
    """Executes the evaluation loop."""
    config = FLAGS.config
    strategy, batch_size = train_utils.setup_strategy(
        config, FLAGS.master,
        FLAGS.devices_per_worker, FLAGS.mode, FLAGS.accelerator_type)

    def input_function(_=None):
        return datasets.get_dataset(
            name=config.dataset,
            config=config,
            batch_size=config.evaluation_batch_size,
            subset=subset)

    model, optimizer, exponetial_moving_average = train_utils.with_strategy(
        lambda: build(config, batch_size, False), strategy)

    metric_keys = ['loss', 'total_loss']
    # metric_keys += model.metric_keys
    metrics = {}
    for metric_key in metric_keys:
        function = functools.partial(tf.keras.metrics.Mean, metric_key)
        current_metric = train_utils.with_strategy(function, strategy)
        metrics[metric_key] = current_metric

    checkpoints = train_utils.with_strategy(
        lambda: train_utils.create_checkpoint(model, optimizer,
                                              exponetial_moving_average),
        strategy)
    dataset = train_utils.dataset_with_strategy(input_function, strategy)

    def step_function(batch):
        _, extra = loss_on_batch(batch, model, config, training=False)

        for metric_key in metric_keys:
            current_metric = metrics[metric_key]
            current_scalar = extra['scalar'][metric_key]
            current_metric.update_state(current_scalar)

    number_of_examples = config.evaluation_number_of_examples
    evaluation_step = train_utils.step_with_strategy(step_function, strategy)
    check_point_path = None
    wait_max = config.get(
        'eval_checkpoint_wait_secs', config.save_checkpoint_secs * 100)
    is_exponetial_moving_average = True if exponetial_moving_average else False

    eval_summary_dir = os.path.join(
        logdir,
        'eval_{}_summaries_pyk_{}'.format(subset, is_exponetial_moving_average))
    writer = tf.summary.create_file_writer(eval_summary_dir)

    while True:
        check_point_path = train_utils.wait_for_checkpoint(logdir,
                                                           check_point_path,
                                                           wait_max)
        logging.info(check_point_path)
        if check_point_path is None:
            logging.info('Timed out waiting for checkpoint.')
            break

        train_utils.with_strategy(
            lambda: train_utils.restore(model, checkpoints, logdir,
                                        exponetial_moving_average),
            strategy)
        data_iterator = iter(dataset)
        number_of_steps = number_of_examples // batch_size

        for metric_key, metric in metrics.items():
            metric.reset_states()

        logging.info('Starting evaluation.')
        done = False
        for step_numbers_ in range(0, number_of_steps,
                                   FLAGS.steps_per_summaries):
            start_run = time.time()
            for substep_numbers_ in range(min(number_of_steps - step_numbers_,
                                              FLAGS.steps_per_summaries)):
                try:
                    if substep_numbers_ % 10 == 0:
                        logging.info('Step: %d',
                                     (step_numbers_ + substep_numbers_ + 1))
                    evaluation_step(data_iterator)
                except (StopIteration, tf.errors.OutOfRangeError):
                    done = True
                    break
            if done:
                break
            bits_per_dimension = metrics['loss'].result()
            logging.info(
                'Bits/Dimension: %.3f, Speed: %.3f seconds/step, Step: %d/%d',
                bits_per_dimension,
                (time.time() - start_run) / FLAGS.steps_per_summaries,
                step_numbers_ + substep_numbers_ + 1, number_of_steps)

        # logging.info('Final Bits/Dim: %.3f', bits_per_dimension)
        with writer.as_default():
            for metric_key, metric in metrics.items():
                current_scalar = metric.result().numpy()
                tf.summary.scalar(metric_key, current_scalar,
                                  step=optimizer.iterations)


def main(_):
    # check missing folders and create
    check_required_folders()
    # preparing images and indexes from objs
    # if FLAGS.

    prepare()

    logging.info('Logging to %s.', FLAGS.logdir)
    if FLAGS.mode == 'train':
        logging.info('[main] I am the trainer.')
        try:
            train(FLAGS.logdir)
        # During TPU Preemeption, the coordinator hangs with the error below.
        # the exception forces the coordinator to fail, and it will be restarted.
        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            os._exit(os.EX_TEMPFAIL)  # pylint: disable=protected-access
    elif FLAGS.mode.startswith('train'):
        logging.info('[main] I am the trainer.')
        train(os.path.join(FLAGS.logdir, FLAGS.mode))
    elif FLAGS.mode == 'eval_train':
        logging.info('[main] I am the training set evaluator.')
        evaluate(FLAGS.logdir, subset='train')
    elif FLAGS.mode == 'eval_valid':
        logging.info('[main] I am the validation set evaluator.')
        evaluate(FLAGS.logdir, subset='valid')
    elif FLAGS.mode == 'eval_test':
        logging.info('[main] I am the test set evaluator.')
        evaluate(FLAGS.logdir, subset='test')
    else:
        raise ValueError(
            'Unknown mode {}. '
            'Must be one of [train, eval_train, eval_valid, eval_test]'.format(
                FLAGS.mode))


if __name__ == '__main__':
    app.run(main)
