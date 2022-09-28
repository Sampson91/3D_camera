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

"""Wrapper for datasets."""

import functools
import os
import re
import tensorflow as tf
import tensorflow_datasets as tensorflow_datasets
from utils import datasets_utils


def resize_to_square(image, resolution=32, train=True):
    """Preprocess the image in a way that is OK for generative modeling."""

    # Crop a square-shaped image by shortening the longer side.
    image_shape = tf.shape(image)
    height, width, channels = image_shape[0], image_shape[1], image_shape[2]
    side_size = tf.minimum(height, width)
    cropped_shape = tf.stack([side_size, side_size, channels])
    if train:
        image = tf.image.random_crop(image, cropped_shape)
    else:
        image = tf.image.resize_with_crop_or_pad(
            image, target_height=side_size, target_width=side_size)

    image = datasets_utils.change_resolution(image, resolution=resolution,
                                             method='area')
    return image


def preprocess(example, train=True, resolution=256):
    """Apply random crop (or) central crop to the image."""
    image = example

    is_label = False
    if isinstance(example, dict):  # isinstance->builtins.py 不能改dict
        image = example['image']
        is_label = 'label' in example.keys()

    image = resize_to_square(image, train=train, resolution=resolution)

    # keepng 'file_name' key creates some undebuggable TPU Error.
    example_copy = dict()
    example_copy['image'] = image
    example_copy['targets'] = image
    if is_label:
        example_copy['label'] = example['label']
    return example_copy


def get_generated_dataset(data_direction, batch_size):
    """Converts a list of generated TFRecords into a TF Dataset."""

    def parse_example(example_proto, resolution=64):
        features = {
            'image': tf.io.FixedLenFeature([resolution * resolution * 3],
                                           tf.int64)
        }
        example = tf.io.parse_example(example_proto, features=features)
        image = tf.reshape(example['image'], (resolution, resolution, 3))
        return {'targets': image}

    # Provided generated dataset.
    def tf_record_name_to_number(generated_dataset):
        generated_dataset = generated_dataset.split('.')[0]
        generated_dataset = re.split(r'(\d+)', generated_dataset)  # re 库函数
        return int(generated_dataset[1])

    assert data_direction is not None
    records = tf.io.gfile.listdir(data_direction)
    max_number = max(records, key=tf_record_name_to_number)
    max_number = tf_record_name_to_number(max_number)

    records = []
    for record in range(max_number + 1):
        path = os.path.join(data_direction, f'gen{record}.tfrecords')
        records.append(path)

    tf_dataset = tf.data.TFRecordDataset(records)
    tf_dataset = tf_dataset.map(parse_example, num_parallel_calls=100)
    tf_dataset = tf_dataset.batch(batch_size=batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    return tf_dataset


def create_generated_dataset_from_images(image_direction):
    """Creates a dataset from the provided directory."""

    def load_image(path):
        image_str = tf.io.read_file(path)
        return tf.image.decode_image(image_str, channels=3)

    child_files = tf.io.gfile.listdir(image_direction)
    files = [os.path.join(image_direction, file) for file in child_files]
    files = tf.convert_to_tensor(files, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((files))
    return dataset.map(load_image, num_parallel_calls=100)


def get_imagenet(subset, read_config):
    """Gets imagenet dataset."""
    train = subset == 'train'
    number_of_validation_examples = 0 if subset == 'eval_train' else 10000
    if subset == 'test':
        datasets = tensorflow_datasets.load('imagenet2012', split='validation',
                                            shuffle_files=False)
    else:
        # split 10000 samples from the imagenet dataset for validation.
        datasets, info = tensorflow_datasets.load('imagenet2012', split='train',
                                                  with_info=True,
                                                  shuffle_files=train,
                                                  read_config=read_config)
        number_of_train = info.splits[
                              'train'].num_examples - number_of_validation_examples
        if train:
            datasets = datasets.take(number_of_train)
        elif subset == 'valid':
            datasets = datasets.skip(number_of_train)
    return datasets


def get_dataset(name,
                config,
                batch_size,
                subset,
                read_config=None,
                data_direction=None):
    """Wrapper around TF-Datasets.

    * Setting `config.random_channel to be True` adds
      datasets['targets_slice'] - Channel picked at random. (of 3).
      datasets['channel_index'] - Index of the randomly picked channel
    * Setting `config.downsample` to be True, adds:.
      datasets['targets_64'] - Downsampled 64x64 input using tf.resize.
      datasets['targets_64_up_back] - 'targets_64' upsampled using tf.resize

    Args:
      name: imagenet
      config: dict
      batch_size: batch size.
      subset: 'train', 'eval_train', 'valid' or 'test'.
      read_config: optional, tfds.ReadConfg instance. This is used for sharding
                   across multiple workers.
      data_direction: Data Directory, Used for Custom dataset.
    Returns:
     dataset: TF Dataset.
    """
    downsample = config.get('downsample', False)
    random_channel = config.get('random_channel', False)
    downsample_res = config.get('downsample_res', 64)
    downsample_method = config.get('downsample_method', 'area')
    num_epochs = config.get('num_epochs', -1)
    data_direction = config.get('data_dir') or data_direction
    auto = tf.data.AUTOTUNE
    train = subset == 'train'

    if name == 'imagenet':
        datasets = get_imagenet(subset, read_config)
    elif name == 'custom':
        assert data_direction is not None
        datasets = create_generated_dataset_from_images(data_direction)
    else:
        raise ValueError(f'Expected dataset in [imagenet, custom]. Got {name}')

    datasets = datasets.map(
        lambda x: preprocess(x, train=train), num_parallel_calls=100)
    if train and random_channel:
        datasets = datasets.map(datasets_utils.random_channel_slice)
    if downsample:
        downsample_part = functools.partial(
            datasets_utils.downsample_and_upsample,
            train=train,
            downsample_res=downsample_res,
            upsample_res=256,
            method=downsample_method)
        datasets = datasets.map(downsample_part, num_parallel_calls=100)

    if train:
        datasets = datasets.repeat(num_epochs)
        datasets = datasets.shuffle(buffer_size=128)
    datasets = datasets.batch(batch_size, drop_remainder=True)
    datasets = datasets.prefetch(auto)
    return datasets
