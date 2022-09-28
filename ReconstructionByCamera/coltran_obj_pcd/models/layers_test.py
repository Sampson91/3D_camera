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

"""Tests for coltran layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
from models import layers

layer_hyper_parameters = itertools.product(["mean", "learnable"],
                                           ["sc", "cs"])
layer_hyper_parameters = [(row1 + row2, row1, row2) for row1, row2 in
                          layer_hyper_parameters]


class LayersTest(tf.test.TestCase, parameterized.TestCase):

    def test_cache_layer(self):
        cache = layers.Cache(canvas_shape=(2, 4))

        # update 1
        experience_first = tf.range(8, dtype=tf.float32)
        experience_first = tf.reshape(experience_first, (1, 2, 2, 2))
        index = tf.stack([0, 0])
        out = cache(inputs=(experience_first, index))
        out_slice = out.numpy()[:1, :2, :2, :2]
        self.assertTrue(np.allclose(out_slice, experience_first.numpy()))

        # update 2
        experience_second = tf.range(8, 16, dtype=tf.float32)
        experience_second = tf.reshape(experience_second, (1, 2, 2, 2))
        index = tf.stack([0, 2])
        out = cache(inputs=(experience_second, index))
        out_numpy = out.numpy()
        first, second = out_numpy[:1, :2, :2, :2], out_numpy[:1, :2, 2:, :2]
        self.assertTrue(np.allclose(second, experience_second.numpy()))
        self.assertTrue(np.allclose(first, experience_first.numpy()))

        # update 3 (special case)
        experience_third = tf.reshape([50.0, 100.0], (1, 1, 1, 2))
        index = tf.stack([0, 0])
        out = cache(inputs=(experience_third, index))
        out_numpy = out.numpy()
        self.assertTrue(np.allclose(out_numpy[0, 0, 0, :2], [50.0, 100.0]))

    def test_shift_layer(self):
        # shift down
        down_shift = layers.Shift(dimension=0, resolution=[3, 3])
        input_numpy = np.arange(9).reshape((1, 3, 3))
        input_tensor = tf.convert_to_tensor(input_numpy)
        input_down = down_shift(input_tensor).numpy()
        equality = input_numpy[:, :-1] == input_down[:, 1:]
        self.assertTrue(np.all(equality))

        # shift right.
        right_shift = layers.Shift(dimension=1, resolution=[3, 3])
        input_numpy = np.arange(9).reshape((1, 3, 3))
        input_tensor = tf.convert_to_tensor(input_numpy)
        input_right = right_shift(input_tensor).numpy()
        equality = input_numpy[:, :, :-1] == input_right[:, :, 1:]
        self.assertTrue(np.all(equality))

    def test_position_embed(self):
        position_embed = layers.PositionEmbed(
            axes=[1, 2], max_lengths=[64, 32])
        inputs = tf.random.uniform(shape=(8, 64, 32, 256))
        embedded = position_embed(inputs)
        for variable in position_embed.variables:
            if len(variable.shape) == 3:
                self.assertEqual(variable.shape, (64, 1, 256))
            else:
                self.assertEqual(variable.shape, (32, 256))
        self.assertEqual(embedded.shape, (8, 64, 32, 256))

    @parameterized.named_parameters(*layer_hyper_parameters)
    def test_conditional_layer_norm(self, spatial_average, sequence):
        condition_layer_normalization = layers.ConditionalLayerNormalization(
            spatial_average=spatial_average, sequence=sequence)
        inputs = tf.random.uniform(shape=(8, 32, 32, 128))
        condition_inputs = tf.random.uniform(shape=(8, 32, 32, 128))
        out = condition_layer_normalization(inputs=(inputs, condition_inputs))
        self.assertEqual(out.shape, (8, 32, 32, 128))

    def test_self_attention_number_dimension_condition_scale(self):
        row_mask = layers.SelfAttentionNumberDimension(
            hidden_size=256, number_of_heads=4,
            number_dimension_block_size=[1, 32],
            resolution=[32, 32], conditional_query=True, conditional_key=True,
            conditional_value=True,
            conditional_scale=True)
        inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
        condition_inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
        output = row_mask(inputs=(inputs, condition_inputs))
        self.assertEqual(output.shape, (1, 3, 32, 32, 256))

    def test_self_attention_number_dimension_condition_scale_false(self):
        row_mask = layers.SelfAttentionNumberDimension(
            hidden_size=256, number_of_heads=4,
            number_dimension_block_size=[1, 32],
            resolution=[32, 32], conditional_query=True, conditional_key=True,
            conditional_value=True,
            conditional_scale=False)
        inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
        condition_inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
        output = row_mask(inputs=(inputs, condition_inputs))
        self.assertEqual(output.shape, (1, 3, 32, 32, 256))

    def test_row_attention(self):
        # row with cache
        row = layers.SelfAttentionNumberDimension(
            hidden_size=256, number_of_heads=4,
            number_dimension_block_size=[1, 32],
            resolution=[32, 32])
        inputs = tf.random.uniform(shape=[4, 2, 32, 3])
        output = row(inputs=inputs)
        self.assertEqual(row.attention_dimension_query, -3)
        self.assertEqual(row.attention_dimension_key, -3)
        self.assertEqual(output.shape, (4, 2, 32, 256))

    def test_column_attention(self):
        # row with cache
        column = layers.SelfAttentionNumberDimension(
            hidden_size=256, number_of_heads=4,
            number_dimension_block_size=[32, 1],
            resolution=[32, 32])
        inputs = tf.random.uniform(shape=[4, 32, 2, 3])
        output = column(inputs=inputs)
        self.assertEqual(output.shape, (4, 32, 2, 256))

    def test_row_attention_mask(self):
        row_mask = layers.SelfAttentionNumberDimension(
            hidden_size=256, number_of_heads=4,
            number_dimension_block_size=[1, 32],
            resolution=[32, 32], mask="future")
        inputs = tf.random.uniform(shape=[4, 2, 32, 3])
        output = row_mask(inputs=inputs)
        self.assertEqual(row_mask.attention_dimension_key, -3)
        self.assertEqual(row_mask.attention_dimension_query, -3)
        self.assertEqual(output.shape, (4, 2, 32, 256))

    def test_col_attention_mask(self):
        column_mask = layers.SelfAttentionNumberDimension(
            hidden_size=256, number_of_heads=8,
            number_dimension_block_size=[4, 1],
            resolution=[4, 4], mask="future")
        inputs = tf.random.uniform(shape=[4, 4, 2, 3])
        output = column_mask(inputs=inputs)
        self.assertEqual(output.shape, (4, 4, 2, 256))
        self.assertEqual(column_mask.attention_dimension_key, -4)
        self.assertEqual(column_mask.attention_dimension_query, -4)

    def test_factorized_attention(self):
        config = ConfigDict()
        config.hidden_size = 256
        config.first_frame_size = 256
        config.number_of_encoder_layers = 2
        config.number_of_heads = 2
        fact = layers.FactorizedAttention(config)
        inputs = tf.random.uniform(shape=(8, 8, 8, 256))
        output = fact(inputs)
        self.assertEqual(output.shape, (8, 8, 8, 256))


if __name__ == "__main__":
    tf.test.main()
