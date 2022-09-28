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

"""Core components of the colorization transfomer.

Consists of:

1. Grayscale Encoder.
2. Outer Decoder.
3. Inner Decoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from models import layers as coltran_layers
from utils import base_utils


def condition_with_context(inputs, condition_layer, context, condition_type,
                           condition_act):
    condition_act_function = base_utils.act_to_function(condition_act)
    condition_out = condition_layer(context)
    if condition_type == 'shift':
        inputs += condition_out
    elif condition_type == 'affine':
        shift, scale = tf.split(condition_out, num_or_size_splits=2, axis=-1)
        inputs *= condition_act_function(scale)
        inputs += condition_act_function(shift)
    return inputs


def get_position_embeddings(position_embed, inputs_shape):
    embeddings = tf.zeros(shape=inputs_shape)
    return position_embed(embeddings)


class GrayScaleEncoder(layers.Layer):
    """Encodes grayscale version of the image into a 2-D spatial context.

    Consists of a stack of row/column attention layers.
    """

    def __init__(self, config, **kwargs):
        super(GrayScaleEncoder, self).__init__(**kwargs)
        self.config = config
        self.dropout = config.get('dropout', 0.0)

    def build(self, input_shapes):
        self.embedding = layers.Dense(units=self.config.hidden_size)
        self.encoder = coltran_layers.FactorizedAttention(self.config)

    def call(self, inputs):
        if len(inputs.shape) == 4:
            if inputs.shape[-1] != 1:
                raise ValueError('Expected inputs is a grayscale image')
            grayscale = tf.squeeze(inputs, axis=-1)
        grayscale = tf.one_hot(grayscale, depth=256)
        hot_gray = self.embedding(grayscale)
        return self.encoder(hot_gray)


class OuterDecoder(layers.Layer):
    """Outer Decoder with optional conditioning.

    Contains the following sequence of operations:
      1. Positional Embeddings.
      2. (Unmasked Row + Masked Column) self attention * num_layers.
      3. Shift Down (to preserve causal ordering)

    The input is a tuple of 2 arguments (X, h) where h is the conditioning
    input. Transforms the input X into 2-D spatial context C (H, W, D)
    conditioned on h. Each location C[i, j] is a vector of size D that
    summarizes information from X[:i] and h.

    The conditional components can be activated by setting the corresponding
    conditional arguments to True.
      1. Conditional Layer Norm: config.cond_ln
      2. Conditional Self Attention: config.cond_att_k, config.cond_att_q,
                                     config.cond_att_v, config.cond_att_scale.
      3. Conditional MLP: config.cond_mlp
    """

    def __init__(self, config, **kwargs):
        super(OuterDecoder, self).__init__(**kwargs)
        self.config = config
        self.dropout = self.config.get('dropout', 0.0)
        self.skip = self.config.get('skip', True)

        # Conditional MLP
        self.condition_multilayer_perceptron = self.config.get('cond_mlp',
                                                               'affine')
        self.condition_multilayer_perceptron_act = self.config.get(
            'cond_mlp_act', 'identity')

        # Conditional Layer Norm.
        self.condition_layer_normalization = self.config.get('cond_ln', True)
        self.condition_layer_normalization_act = self.config.get('cond_ln_act',
                                                                 'identity')
        self.condition_layer_normalization_sequence = self.config.get(
            'cond_ln_seq', 'sc')
        self.condition_layer_normalization_spatial_pooling_average = self.config.get(
            'cond_ln_sp_ave', 'learnable')
        self.condition_layer_normalization_initial = self.config.get(
            'cond_ln_init', 'glorot_uniform')

        # Conditional Self Attention.
        self.condition_attention_act = self.config.get('cond_att_act',
                                                      'identity')
        self.condition_attention_key = self.config.get('cond_att_k', True)
        self.condition_attention_query = self.config.get('cond_att_q', True)
        self.condition_attention_value = self.config.get('cond_att_v', True)
        self.condition_attention_scale = self.config.get('cond_att_scale', True)
        self.condition_attention_initial = self.config.get('cond_att_init',
                                                          'glorot_uniform')
        self.condition_attention = (self.condition_attention_value
                                    or self.condition_attention_query
                                    or self.condition_attention_key)

    def build(self, input_shapes):
        embed_shape = input_shapes[0]
        height, width, number_of_filters = embed_shape[1:]
        hidden_size = self.config.hidden_size
        number_of_heads = self.config.number_of_heads
        first_frame_size = self.config.first_frame_size
        resolution = [height, width]

        self.position_embed = coltran_layers.PositionEmbed(axes=[1, 2],
                                                           max_lengths=resolution)

        self.residual_layers, self.layer_normalizations, self.conditional_multilayer_perceptron_layers = [], [], []
        # conditional_multilayer_perceptron == cmlp
        number_of_normalizations = self.config.number_of_outer_layers * 4
        if self.condition_layer_normalization:
            for _ in range(number_of_normalizations):
                current_normalization = coltran_layers.ConditionalLayerNormalization(
                    spatial_average=self.condition_layer_normalization_spatial_pooling_average,
                    sequence=self.condition_layer_normalization_sequence,
                    out_initial=self.condition_layer_normalization_initial,
                    out_act=self.condition_layer_normalization_act)
                self.layer_normalizations.append(current_normalization)
        else:
            self.layer_normalizations = [layers.LayerNormalization() for _ in
                                         range(number_of_normalizations)]

        for layer_index in range(self.config.number_of_outer_layers):
            # unmasked row
            unmask_row = coltran_layers.SelfAttentionNumberDimension(
                hidden_size=hidden_size, number_of_heads=number_of_heads,
                number_dimension_block_size=[1, width],
                resolution=[height, width],
                conditional_query=self.condition_attention_query,
                conditional_key=self.condition_attention_key,
                conditional_value=self.condition_attention_value,
                conditional_initial=self.condition_attention_initial,
                conditional_scale=self.condition_attention_scale,
                conditonal_act=self.condition_attention_act,
                name='unmask_row_attention_%d' % layer_index)

            first_frame_row = tf.keras.Sequential([
                layers.Dense(units=first_frame_size, activation='relu'),
                layers.Dense(units=number_of_filters)
            ], name='row_dense_%d' % layer_index)

            # masked column,
            mask_column = coltran_layers.SelfAttentionNumberDimension(
                hidden_size=hidden_size, number_of_heads=number_of_heads,
                mask='future',
                number_dimension_block_size=[height, 1],
                resolution=[height, width],
                conditional_query=self.condition_attention_query,
                conditional_key=self.condition_attention_key,
                conditional_value=self.condition_attention_value,
                conditonal_act=self.condition_attention_act,
                conditional_initial=self.condition_attention_initial,
                conditional_scale=self.condition_attention_scale,
                name='mask_col_att_%d' % layer_index)

            first_frame_column = tf.keras.Sequential([
                layers.Dense(units=first_frame_size, activation='relu'),
                layers.Dense(units=number_of_filters)
            ], name='col_dense_%d' % layer_index)

            self.residual_layers.append(unmask_row)
            self.residual_layers.append(first_frame_row)
            self.residual_layers.append(mask_column)
            self.residual_layers.append(first_frame_column)

            # Conditional MLP layers.
            if self.condition_multilayer_perceptron == 'shift':
                shift_roll = layers.Dense(units=hidden_size,
                                          name='shift_roll_%d' % layer_index)
                shift_column = layers.Dense(units=hidden_size,
                                            name='shift_column_%d' % layer_index)
                self.conditional_multilayer_perceptron_layers.append(shift_roll)
                self.conditional_multilayer_perceptron_layers.append(
                    shift_column)
            elif self.condition_multilayer_perceptron == 'affine':
                affine_row = layers.Dense(
                    units=2 * hidden_size, name='affine_row_%d' % layer_index)
                affine_column = layers.Dense(
                    units=2 * hidden_size,
                    name='affine_column_%d' % layer_index)
                self.conditional_multilayer_perceptron_layers.append(affine_row)
                self.conditional_multilayer_perceptron_layers.append(
                    affine_column)

        self.shift_down = coltran_layers.Shift(dimension=0,
                                               resolution=resolution)

    def call(self, inputs, training=True):
        embeddings, channel_context = inputs
        condition_layer_index = 0

        output = self.position_embed(embeddings)
        if self.skip:
            output += channel_context
        inputs = output

        for layer, normalization in zip(self.residual_layers,
                                        self.layer_normalizations):
            if 'att' in layer.name and self.condition_attention:
                output = layer((inputs, channel_context))
            else:
                output = layer(inputs)

            if 'dense' in layer.name and self.condition_multilayer_perceptron:
                current_conditional_layer = \
                    self.conditional_multilayer_perceptron_layers[
                        condition_layer_index]
                output = condition_with_context(output,
                                                current_conditional_layer,
                                                channel_context,
                                                self.condition_multilayer_perceptron,
                                                self.condition_multilayer_perceptron_act)
                condition_layer_index += 1

            output = coltran_layers.residual_dropout(
                inputs, output, self.dropout, training)

            if self.condition_layer_normalization:
                inputs = normalization((output, channel_context))
            else:
                inputs = normalization(output)

        output = self.shift_down(inputs)
        return output


class InnerDecoder(layers.Layer):
    """Inner Decoder with optional conditioning.

    Contains the following sequence of operations:
      1. Adds positional Embeddings + context to the pixel embeddings.
      2. Shift right (to preserve causal order).
      2. (Masked Row) self attention * num_layers.

    The input is a tuple of 2 arguments (X, h_out, h) where h_out and h are the
    conditioning inputs from the grayscale image and the outer decoder
    respectively. Transforms the input X into 2-D spatial context C (H, W, D)
    conditioned on h. Each location C[i, j] is a vector of size D that
    summarizes information from X[:i], X[i, :j] and h.

    The conditional components can be activated by setting the corresponding
    conditional arguments to True.
      1. Conditional Layer Norm: config.cond_ln
      2. Conditional Self Attention: config.cond_att_k, config.cond_att_q,
                                     config.cond_att_v, config.cond_att_scale.
      3. Conditional MLP: config.cond_mlp
    """

    def __init__(self,
                 config,
                 **kwargs):
        super(InnerDecoder, self).__init__(**kwargs)
        self.config = config
        self.skip = self.config.get('skip', True)
        self.dropout = self.config.get('dropout', 0.0)

        self.conditional_multilayer_perceptron = self.config.get('cond_mlp',
                                                                 'affine')
        self.conditional_multilayer_perceptron_act = self.config.get(
            'cond_mlp_act', 'identity')

        self.condition_layer_normalization = self.config.get('cond_ln', True)
        self.condition_layer_normalization_act = self.config.get('cond_ln_act',
                                                                 'identity')
        self.condition_layer_normalization_sequence = self.config.get(
            'cond_ln_seq', 'sc')
        self.condition_layer_normalization_spatial_pooling_average = self.config.get(
            'cond_ln_sp_ave', 'learnable')
        self.condition_layer_normalization_initial = self.config.get(
            'cond_ln_init', 'glorot_uniform')

        self.conditional_attention_act = self.config.get('cond_att_act',
                                                         'identity')
        self.conditional_attention_key = self.config.get('cond_att_k', False)
        self.conditional_attention_query = self.config.get('cond_att_q', False)
        self.conditional_attention_value = self.config.get('cond_att_v', False)
        self.conditional_attention_scale = self.config.get('cond_att_scale',
                                                           False)
        self.conditional_attention_initial = self.config.get('cond_att_init',
                                                             'glorot_uniform')
        self.conditional_attention = (self.conditional_attention_value
                                      or self.conditional_attention_query
                                      or self.conditional_attention_key)

    def build(self, input_shapes):
        context_shape = input_shapes[1]
        height, width = context_shape[1:3]
        first_frame_size = self.config.first_frame_size
        hidden_size = self.config.hidden_size
        number_of_heads = self.config.number_of_heads
        resolution = [height, width]

        self.position_embed = coltran_layers.PositionEmbed(axes=[1, 2],
                                                           max_lengths=resolution)
        self.shift_right = coltran_layers.Shift(dimension=1,
                                                resolution=resolution)

        self.residual_layers, self.layer_normalizations, self.conditional_multilayer_perceptron_layers = [], [], []
        number_of_normalizations = 2 * self.config.number_of_inner_layers
        if self.condition_layer_normalization:
            for _ in range(number_of_normalizations):
                current_normalization = coltran_layers.ConditionalLayerNormalization(
                    spatial_average=self.condition_layer_normalization_spatial_pooling_average,
                    sequence=self.condition_layer_normalization_sequence,
                    out_initial=self.condition_layer_normalization_initial,
                    out_act=self.condition_layer_normalization_act)
                self.layer_normalizations.append(current_normalization)
        else:
            self.layer_normalizations = [layers.LayerNormalization() for _ in
                                         range(number_of_normalizations)]

        for layer_index_ in range(self.config.number_of_inner_layers):

            mask_row = coltran_layers.SelfAttentionNumberDimension(
                hidden_size=hidden_size, number_of_heads=number_of_heads,
                mask='future',
                number_dimension_block_size=[1, width],
                resolution=[height, width],
                conditional_query=self.conditional_attention_query,
                conditional_key=self.conditional_attention_key,
                conditional_value=self.conditional_attention_value,
                conditional_initial=self.conditional_attention_initial,
                conditional_scale=self.conditional_attention_scale,
                conditonal_act=self.conditional_attention_act,
                name='mask_row_attention_%d' % layer_index_)

            first_frame_block = tf.keras.Sequential([
                layers.Dense(units=first_frame_size, activation='relu'),
                layers.Dense(units=hidden_size)
            ], name='dense_%d' % layer_index_)

            self.residual_layers.append(mask_row)
            self.residual_layers.append(first_frame_block)

            if self.conditional_multilayer_perceptron == 'shift':
                shift_column = layers.Dense(units=hidden_size,
                                            name='shift_c_%d' % layer_index_)
                self.conditional_multilayer_perceptron_layers.append(
                    shift_column)
            elif self.conditional_multilayer_perceptron == 'affine':
                affine_column = layers.Dense(
                    units=2 * hidden_size,
                    name='affine_column_%d' % layer_index_)
                self.conditional_multilayer_perceptron_layers.append(
                    affine_column)

    def call(self, inputs, row_index=None, training=True):
        embeddings, upper_context, channel_context = inputs

        embeddings = self.shift_right(embeddings)
        if row_index is None:
            embeddings = self.position_embed(embeddings)
        # special case during sampling.
        else:
            input_shape = embeddings.shape.as_list()
            position_embed = get_position_embeddings(self.position_embed,
                                                     input_shape)
            position_embed = position_embed[:, row_index: row_index + 1]
            embeddings += position_embed

        inputs = embeddings
        if self.skip:
            inputs += channel_context
            inputs += upper_context

        layer_zip = zip(self.residual_layers, self.layer_normalizations)
        all_context = tf.concat((channel_context, upper_context), -1)

        condition_layer_index = 0
        for layer, normalization in layer_zip:

            # Conditional Self-Attention.
            if 'att' in layer.name and self.conditional_attention:
                output = layer((inputs, all_context))
            else:
                output = layer(inputs)

            # Conditional MLP.
            if 'dense' in layer.name and self.conditional_multilayer_perceptron:
                current_conditional_layer = \
                    self.conditional_multilayer_perceptron_layers[
                        condition_layer_index]
                output = condition_with_context(output,
                                                current_conditional_layer,
                                                all_context,
                                                self.conditional_multilayer_perceptron,
                                                self.conditional_multilayer_perceptron_act)
                condition_layer_index += 1

            output = coltran_layers.residual_dropout(
                inputs, output, self.dropout, training)

            # providing all context here violates causal masking due to the spatial
            # averaging.
            # Conditional Layer normalization.
            if self.condition_layer_normalization:
                inputs = normalization((output, channel_context))
            else:
                inputs = normalization(output)

        return inputs
