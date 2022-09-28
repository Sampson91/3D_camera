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

"""Tests for base_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf
from utils import base_utils


class UtilsTest(tf.test.TestCase):

    def test_quantize(self):
        tensorflow_range = tf.range(0, 256, dtype=tf.int32)
        actual = base_utils.convert_bits(tensorflow_range, number_of_bits_in=8,
                                         number_of_bits_out=5).numpy()
        expected = np.repeat(np.arange(0, 32), 8)
        self.assertTrue(np.allclose(expected, actual))

    def test_dequantize(self):
        tensorflow_range = tf.range(0, 32, dtype=tf.int32)
        actual = base_utils.convert_bits(tensorflow_range, number_of_bits_in=5,
                                         number_of_bits_out=8).numpy()
        expected = np.arange(0, 256, 8)
        self.assertTrue(np.allclose(expected, actual))

    def test_rgb_to_ycbcr(self):
        tensorflow_random_uniform = tf.random.uniform(shape=(2, 32, 32, 3))
        ycbcr = base_utils.rgb_to_ycbcr(tensorflow_random_uniform)
        self.assertEqual(ycbcr.shape, (2, 32, 32, 3))

    def test_image_hist_to_bit(self):
        tensorflow_random_uniform = tf.random.uniform(shape=(2, 32, 32, 3),
                                                      minval=0, maxval=256,
                                                      dtype=tf.int32)
        hist = base_utils.image_to_hist(tensorflow_random_uniform,
                                        num_symbols=256)
        self.assertEqual(hist.shape, (2, 3, 256))

    def test_labels_to_bins(self):
        number_of_bits = 3
        bins = np.arange(2 ** number_of_bits)
        triplets = itertools.product(bins, bins, bins)

        labels = np.array(list(triplets))
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        bins_tensor = base_utils.labels_to_bins(labels_tensor,
                                                number_of_symbols_per_channel=8)
        bins_numpy = bins_tensor.numpy()
        self.assertTrue(np.allclose(bins_numpy, np.arange(512)))

        inverse_labels_tensor = base_utils.bins_to_labels(bins_tensor,
                                                          number_of_symbols_per_channel=8)
        inverse_labels_numpy = inverse_labels_tensor.numpy()
        self.assertTrue(np.allclose(labels, inverse_labels_numpy))

    def test_bins_to_labels_random(self):
        labels_tensor = tf.random.uniform(shape=(1, 64, 64, 3), minval=0,
                                          maxval=8,
                                          dtype=tf.int32)
        labels_numpy = labels_tensor.numpy()
        bins_tensor = base_utils.labels_to_bins(labels_tensor,
                                                number_of_symbols_per_channel=8)

        inverse_labels_tensor = base_utils.bins_to_labels(bins_tensor,
                                                          number_of_symbols_per_channel=8)
        inverse_labels_numpy = inverse_labels_tensor.numpy()
        self.assertTrue(np.allclose(inverse_labels_numpy, labels_numpy))


if __name__ == '__main__':
    tf.test.main()
