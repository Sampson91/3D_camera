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

"""Utils for training."""

import os
import time
from absl import logging
import numpy as np
import tensorflow as tf
import yaml


def step_with_strategy(step_function, strategy):
    def _step(iterator):
        if strategy is None:
            step_function(next(iterator))
        else:
            strategy.experimental_run(step_function, iterator)

    return _step


def write_config(config, logdir):
    """Write config dict to a directory."""
    tf.io.gfile.makedirs(logdir)
    with tf.io.gfile.GFile(os.path.join(logdir, 'config.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f)


def wait_for_checkpoint(observe_directions, previous_path=None, max_wait=-1):
    """Returns new checkpoint paths, or None if timing out."""
    is_single = isinstance(observe_directions, str)
    if is_single:
        observe_directions = [observe_directions]
        if previous_path:
            previous_path = [previous_path]

    start_time = time.time()
    previous_path = previous_path or [None for _ in observe_directions]
    while True:
        new_path = [tf.train.latest_checkpoint(d) for d in observe_directions]
        if all(a != b for a, b in zip(new_path, previous_path)):
            if is_single:
                latest_check_point = new_path[0]
            else:
                latest_check_point = new_path
            if latest_check_point is not None:
                return latest_check_point
        if max_wait > 0 and (time.time() - start_time) > max_wait:
            return None
        logging.info('Sleeping 60s, waiting for checkpoint.')
        time.sleep(60)


def build_optimizer(config):
    """Builds optimizer."""
    optimizer_config = dict(config.optimizer)
    optimizer_type = optimizer_config.pop('type', 'rmsprop')
    if optimizer_type == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(**optimizer_config)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(**optimizer_config)
    elif optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(**optimizer_config)
    else:
        raise ValueError('Unknown optimizer %s.' % optimizer_type)
    return optimizer


def build_exponetial_moving_average(config,
                                    exponetial_moving_average_variables):
    """Builds exponential moving average."""
    exponetial_moving_average = None
    polyak_decay = config.get('polyak_decay', 0.0)
    if polyak_decay:
        exponetial_moving_average = tf.train.ExponentialMovingAverage(
            polyak_decay)
        exponetial_moving_average.apply(exponetial_moving_average_variables)
        logging.info('Built with exponential moving average.')
    return exponetial_moving_average


def setup_strategy(config, master, devices_per_worker, mode, accelerator_type):
    """Set up strategy."""
    if accelerator_type == 'TPU':
        cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=master)
        tf.config.experimental_connect_to_cluster(cluster)
        topology = tf.tpu.experimental.initialize_tpu_system(cluster)
        strategy = tf.distribute.experimental.TPUStrategy(cluster)
        number_of_cores = topology.num_tasks * topology.num_tpus_per_task
    else:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = None
        number_of_cores = devices_per_worker

    tpu_batch_size = config.get('eval_batch_size', 0)
    if mode.startswith('train') or not tpu_batch_size:
        tpu_batch_size = config.batch_size
    tpu_batch_size *= number_of_cores
    logging.info('Running on %d number of cores with total batch_size of %d.',
                 number_of_cores, tpu_batch_size)

    return strategy, tpu_batch_size


def dataset_with_strategy(dataset_function, strategy):
    if strategy:
        return strategy.experimental_distribute_datasets_from_function(
            dataset_function)
    else:
        return dataset_function(None)


def with_strategy(function, strategy):
    logging.info(strategy)
    if strategy:
        with strategy.scope():
            return function()
    else:
        return function()


def create_checkpoint(models, optimizer=None, exponetial_moving_average=None,
                      scope=None):
    """Creates tf.train.Checkpoint instance."""
    single_model = not isinstance(models, (tuple, list))
    checkpoints = []
    for model_ in [models] if single_model else models:
        exponetial_moving_average_variables = get_exponetial_moving_average_vars(
            exponetial_moving_average, model_)
        if filter is None:
            to_save = {variable_.name: variable_ for variable_ in
                       model_.variables if scope in variable_.name}
        else:
            to_save = {variable_.name: variable_ for variable_ in
                       model_.variables}
        to_save.update(exponetial_moving_average_variables)
        if optimizer is not None and scope is None:
            to_save['optimizer'] = optimizer
        checkpoints.append(
            tf.train.Checkpoint(**to_save))
    return checkpoints[0] if single_model else checkpoints


def get_curr_step(check_point_path):
    """Parse curr training step from checkpoint path."""
    variable_names = tf.train.list_variables(check_point_path)
    for variable_name_, _ in variable_names:
        if 'iter' in variable_name_:
            step = tf.train.load_variable(check_point_path, variable_name_)
            return step


def get_exponetial_moving_average_vars(exponetial_moving_average, model):
    """Get exponetial_moving_average variables."""
    if exponetial_moving_average:
        try:
            return {
                exponetial_moving_average.average(
                    variable_).name: exponetial_moving_average.average(
                    variable_) for variable_ in model.trainable_variables
            }
        except:  # pylint: disable=bare-except
            exponetial_moving_average.apply(model.trainable_variables)
            return {
                exponetial_moving_average.average(
                    variable_).name: exponetial_moving_average.average(
                    variable_) for variable_ in model.trainable_variables
            }
        else:
            return {}
    else:
        return {}


def restore(model, check_point, check_point_direction,
            exponetial_moving_average=None):
    if not isinstance(model, (tuple, list)):
        model, check_point, check_point_direction = [model], [check_point], [
            check_point_direction]
    for model_, check_point_, check_point_direction_ in zip(model, check_point,
                                                            check_point_direction):
        logging.info('Restoring from %s.', check_point_direction_)
        check_point_.restore(
            tf.train.latest_checkpoint(check_point_direction_)).expect_partial()
        if exponetial_moving_average:
            for variable_ in model_.trainable_variables:
                variable_.assign(exponetial_moving_average.average(variable_))


def save_nparray_to_disk(filename, nparray):
    file_direction, _ = os.path.split(filename)
    if not tf.io.gfile.exists(file_direction):
        tf.io.gfile.makedirs(file_direction)
    with tf.io.gfile.GFile(filename, 'w') as file_:
        np.save(file_, nparray)
