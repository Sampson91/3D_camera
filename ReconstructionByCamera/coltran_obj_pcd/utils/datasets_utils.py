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

"""Utils for datasets loading."""

import tensorflow as tf


def change_resolution(image, resolution, method='area'):
    image = tf.image.resize(image, method=method, antialias=True,
                            size=(resolution, resolution))
    image = tf.cast(tf.round(image), dtype=tf.int32)
    return image


def downsample_and_upsample(sample, train, downsample_res, upsample_res,
                            method):
    # 这里的downsample_res, upsample_res需要进入 tf__downsample_and_upsample()

    """Downsample and upsample."""
    keys = ['targets']
    if train and 'targets_slice' in sample.keys():
        keys += ['targets_slice']

    for key in keys:
        inputs = sample[key]
        # Conditional low resolution input.
        sample_down = change_resolution(inputs, resolution=downsample_res,
                                        method=method)
        sample['%s_%d' % (key, downsample_res)] = sample_down

        # We upsample here instead of in the model code because some upsampling
        # methods are not TPU friendly.
        sample_up = change_resolution(sample_down, resolution=upsample_res,
                                      method=method)
        sample['%s_%d_up_back' % (key, downsample_res)] = sample_up
    return sample


def random_channel_slice(random_slice):
    random_channel = tf.random.uniform(
        shape=[], minval=0, maxval=3, dtype=tf.int32)
    targets = random_slice['targets']
    resolution = targets.shape[1]
    image_slice = targets[Ellipsis, random_channel: random_channel + 1]
    image_slice.set_shape([resolution, resolution, 1])
    random_slice['targets_slice'] = image_slice
    random_slice['channel_index'] = random_channel
    return random_slice
