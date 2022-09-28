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

"""Tests for core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging
from absl.testing import parameterized
from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
from models import core


def get_number_of_variables(model):
    variable_shapes = [np.prod(variable.shape) for variable in model.variables]
    return np.sum(variable_shapes)


condition_hyper_parameters = itertools.product(["shift", "affine"],
                                               [True, False],
                                               [True, False],
                                               [True, False])

new_hyper_parameters = []
for hyper_parameters in condition_hyper_parameters:
    new_argumentations = "_".join(
        [str(hyper_parameters_) for hyper_parameters_ in hyper_parameters])
    new_hyper_parameters.append((new_argumentations, *hyper_parameters))


class ColTranComponentsTest(tf.test.TestCase, parameterized.TestCase):

    def get_config(self):
        config = ConfigDict()
        config.hidden_size = 256
        config.first_frame_size = 256
        config.image_bit_depth = 5
        config.number_of_symbols = 32
        config.number_of_heads = 4
        config.resolution = [8, 8]
        config.number_of_outer_layers = 1
        config.number_of_inner_layers = 3
        config.number_of_encoder_layers = 1
        config.batch_size = 2
        config.skip = True

        config.conditional_multilayer_perceptron = "affine_dense"
        config.conditional_multilayer_perceptron_act = "identity"

        config.condition_layer_normalization = True
        config.condition_layer_normalization_act = "tanh"
        config.condition_layer_normalization_sequence = "cs"
        config.condition_layer_normalization_spatial_pooling_average = "learnable"
        config.condition_layer_normalization_initial = "glorot_uniform"

        config.conditional_attention_act = "identity"
        config.conditional_attention_scale = True
        config.conditional_attention_key = True
        config.conditional_attention_query = True
        config.conditional_attention_value = True
        return config

    def test_grayscale_encoder(self):
        config = self.get_config()
        inputs = tf.random.uniform(shape=(2, 32, 32, 3), minval=0, maxval=256,
                                   dtype=tf.int32)
        gray = tf.image.rgb_to_grayscale(inputs)
        encoder = core.GrayScaleEncoder(config)
        output = encoder(gray)
        self.assertEqual(output.shape, (2, 32, 32, 256))

    @parameterized.named_parameters(*new_hyper_parameters)
    def test_inner_decoder(self, conditional_multilayer_perceptron,
                           condition_layer_normalization,
                           conditional_attention_query,
                           conditional_attention_scale):
        embeddings = tf.random.uniform(shape=(2, 8, 8, 256))
        channel_context = tf.random.uniform(shape=(2, 8, 8, 256))
        upper_context = tf.random.uniform(shape=(2, 8, 8, 256))
        config = self.get_config()
        config.conditional_multilayer_perceptron = conditional_multilayer_perceptron
        config.condition_layer_normalization = condition_layer_normalization
        config.conditional_attention_query = conditional_attention_query
        config.conditional_attention_scale = conditional_attention_scale

        model = core.InnerDecoder(config=config)
        output = model(inputs=(embeddings, upper_context, channel_context))
        number_of_variables = get_number_of_variables(model)
        logging.info(number_of_variables)
        self.assertEqual(output.shape, (2, 8, 8, 256))

    @parameterized.named_parameters(*new_hyper_parameters)
    def test_outer_decoder(self, conditional_multilayer_perceptron,
                           condition_layer_normalization,
                           conditional_attention_query,
                           conditional_attention_scale):
        embeddings = tf.random.uniform(shape=(2, 8, 8, 256))
        channel_context = tf.random.uniform(shape=(2, 8, 8, 256))
        config = self.get_config()
        config.conditional_multilayer_perceptron = conditional_multilayer_perceptron
        config.condition_layer_normalization = condition_layer_normalization
        config.conditional_attention_query = conditional_attention_query
        config.conditional_attention_scale = conditional_attention_scale

        model = core.OuterDecoder(config=config)
        number_of_variables = get_number_of_variables(model)
        logging.info(number_of_variables)
        upper_context = model(inputs=(embeddings, channel_context))
        upper_context_numpy = upper_context.numpy()

        # the first row slice should have zero context since both the present
        # and future are effectively masked.
        self.assertTrue(np.allclose(upper_context_numpy[:, 0], 0.0))
        self.assertEqual(upper_context_numpy.shape, (2, 8, 8, 256))


if __name__ == "__main__":
    tf.test.main()
