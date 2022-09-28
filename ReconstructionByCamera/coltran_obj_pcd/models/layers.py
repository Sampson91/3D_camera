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

"""Various base layers for the colorization transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import operator
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from utils import att_utils
from utils import base_utils


# pylint: disable=duplicate-string-formatting-argument


def residual_dropout(inputs, output, dropout, training):
    """out = inputs + dropout(output)."""
    if training and dropout:
        output = tf.nn.dropout(output, dropout)
    output += inputs
    return output


class Shift(layers.Layer):
    """Shifts an input tensor either down or right to preserve causal ordering."""

    def __init__(self, dimension, resolution, **kwargs):
        """Init.

        Args:
          dimension: int, 0 to shift down, 1 to shift right.
          resolution: list of 2 ints, [H, W].
          **kwargs:
        """
        super(Shift, self).__init__(**kwargs)
        self.dimension = dimension
        self.resolution = resolution

    def call(self, shift_tensor):
        shape = shift_tensor.shape
        rank = len(shape)
        dimension = self.dimension + 1

        # Assume 1 batch_dim.
        index = [0] * len(self.resolution)
        coppied_shift_tensor = shift_tensor
        paddings = np.zeros((rank, 2), dtype=np.int32)
        paddings[dimension, 0] = 1
        coppied_shift_tensor = tf.pad(coppied_shift_tensor, paddings)

        remain_dimensions = rank - 1 - len(index[:dimension])
        slice_indexes = [0] + index[:dimension] + [0] * remain_dimensions
        return tf.slice(coppied_shift_tensor, slice_indexes, shape)


class Cache(layers.Layer):
    """Keras layer for cacheing.

    Values are cached in a tensor of shape (B, canvas_shape, D).
    B and D are inferred from the inputs to the call method.

    Every call to the cache instance is assumed to be a tuple of (index, values).
    It updates the cache such that cache[:, index:, :] = values
    """

    def __init__(self, canvas_shape,
                 number_of_batch_axes=1,
                 dtype=tf.float32,
                 **kwargs):
        super(Cache, self).__init__(trainable=False, **kwargs)
        self.canvas_shape = canvas_shape
        self.number_of_batch_axes = number_of_batch_axes
        self._dtype = dtype

    def build(self, input_shapes):
        number_of_canvas_dimension = len(self.canvas_shape)
        value, _ = input_shapes
        features_shape = value[
                         self.number_of_batch_axes + number_of_canvas_dimension:]
        cache_shape = (value[:self.number_of_batch_axes] + self.canvas_shape +
                       features_shape)
        self.cache = tf.zeros(shape=cache_shape, dtype=self._dtype)
        super(Cache, self).build(input_shapes)

    def reset(self):
        self.cache = tf.zeros(shape=self.cache.shape, dtype=self._dtype)

    def call(self, inputs):
        value, index = inputs
        if self.cache.shape == inputs[0].shape:
            self.cache = value
            return value

        shape = self.cache.shape.as_list()
        number_of_index_axes = index.shape[0]
        number_of_batch_axes = self.number_of_batch_axes
        number_of_feature_axes = len(
            shape) - number_of_index_axes - number_of_batch_axes
        features_shape = shape[number_of_batch_axes + number_of_index_axes:]
        batch_shape = shape[:number_of_batch_axes]

        value_index_shape = tf.shape(value)[
                            number_of_batch_axes:-number_of_feature_axes]
        if tf.reduce_max(value_index_shape) > 1:
            # This is a block update starting at index.
            value_ranges = []
            for i, shape_ in enumerate(tf.unstack(value_index_shape)):
                current_range = tf.range(index[i], index[i] + shape_)
                value_ranges.append(current_range)

            batch_ranges = [tf.range(shape_) for shape_ in batch_shape]

            mesh = tf.meshgrid(*(batch_ranges + value_ranges), indexing='ij')
            indices = tf.stack(mesh, axis=-1)
            indices = tf.reshape(indices, [-1,
                                           number_of_index_axes + number_of_batch_axes])
        else:
            # This is a single update at index position.
            batch_ranges = [tf.range(shape_) for shape_ in batch_shape]
            mesh = tf.meshgrid(*batch_ranges, indexing='ij')
            batch_indices = tf.stack(mesh, axis=-1)
            batch_indices = tf.reshape(batch_indices,
                                       [-1, number_of_batch_axes])

            # Add leading axes to nd-index and tile to get batched indices.
            shape_indices = tf.reshape(index, [1] * number_of_batch_axes + [-1])
            shape_indices = tf.tile(shape_indices, batch_shape + [1])
            shape_indices = tf.reshape(shape_indices,
                                       [-1, number_of_index_axes])

            indices = tf.concat([batch_indices, shape_indices], axis=-1)

        # We need to squeeze nd-axes from value before updating.
        value = tf.reshape(value, [-1] + features_shape)
        self.cache = tf.tensor_scatter_nd_update(self.cache, indices, value)
        return self.cache


class Masking(object):
    """Masking options for self-attention.

    We can either mask the entire future, i.e. allow looking into the past and
    the current element, or we can mask in addition the present as well, i.e.,
    we can look only to the past.
    """

    FUTURE = 'future'
    FUTURE_PRESENT = 'future_present'


class PositionEmbed(layers.Layer):
    """Adds factorized positional embeddings for specified axes."""

    def __init__(self, axes, max_lengths=None, **kwargs):
        """Init.

        Args:
          axes: list of ints, axis over which to apply the positional embeddings.
          max_lengths: list of ints, maximum length over each axis.
          **kwargs:
        """
        super(PositionEmbed, self).__init__(**kwargs)
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        self.axes = axes
        self.max_lengths = None
        if max_lengths:
            if not isinstance(max_lengths, (list, tuple)):
                max_lengths = [max_lengths]
            self.max_lengths = max_lengths

    def build(self, input_shape):
        rank = len(input_shape)
        self.axes = sorted(
            [rank + axe_ if axe_ < 0 else axe_ for axe_ in self.axes])
        self.max_lengths = self.max_lengths or [input_shape[axe_] for axe_ in
                                                self.axes]
        self.embeddings = []
        for i, axis in enumerate(self.axes):
            shape = [self.max_lengths[i]] + [1] * (rank - axis - 2)
            shape.append(input_shape[-1])
            initializer = tf.keras.initializers.RandomNormal(
                stddev=shape[-1] ** -0.5)
            self.embeddings.append(
                self.add_weight(
                    name='position_embedding_%d' % i,
                    shape=shape,
                    initializer=initializer,
                    trainable=True))
        super(PositionEmbed, self).build(input_shape)

    def call(self, inputs):
        out = inputs
        for embed_ in self.embeddings:
            out += embed_
        return out


class DenseNumberDimension(layers.Layer):
    """Maps a rank-m tensor to a rank-n tensor through a dense contraction."""

    def __init__(self,
                 filters,
                 contract_axes=1,
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(DenseNumberDimension, self).__init__(**kwargs)
        if isinstance(filters, int):
            filters = [filters]
        self.filters = tuple(filters)
        self.contract_axes = contract_axes
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer

        # Behaviours differ when shape(weights) > 2.
        # see: https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/init_ops_v2.py#L733 pylint: disable=line-too-long
        if self._kernel_initializer == 'glorot_uniform_number_dimension':
            self._kernel_initializer = self._glorot_uniform

    def _num_batch_axes(self, input_shape):
        """Returns number of batch axes in inputs."""
        return len(input_shape) - len(self.contract_shape)

    def _glorot_uniform(self, shape, dtype=tf.float32):
        """Glorot uniform initializer."""
        fan_out = functools.reduce(operator.mul, self.filters)
        fan_in = functools.reduce(operator.mul, shape[:self.contract_axes])
        scale = 1. / max(1., (fan_in + fan_out) / 2.)
        limit = math.sqrt(3.0 * scale)
        return tf.random.uniform(shape, -limit, limit, dtype)

    def build(self, input_shape):
        # Infer matrix multiplication if no contract shape specified.
        self.contract_shape = input_shape[-self.contract_axes:]
        wwight_shape = self.contract_shape + self.filters
        self.weight = self.add_weight(
            name='kernel',
            shape=wwight_shape,
            initializer=self._kernel_initializer,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=self.filters,
                initializer=self.bias_initializer,
                trainable=True)
        super(DenseNumberDimension, self).build(input_shape)

    def call(self, inputs):
        # Workaround lack of ellipsis support.
        # pyformat: disable
        number_of_batch_axes = self._num_batch_axes(inputs.shape)
        batch_str = 'abcdefghijklm'[:number_of_batch_axes]
        contract_str = 'ABCDEFGHIJKLM'[:len(self.contract_shape)]
        output_str = 'nopqrstuvwxyz'[:len(self.filters)]
        # pyformat: enable
        einsum_str = '{}{},{}{}->{}{}'.format(batch_str, contract_str,
                                              contract_str,
                                              output_str, batch_str, output_str)
        result = tf.einsum(einsum_str, inputs, self.weight)
        if self.use_bias:
            result += self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class RelativeAttentionBiasNumberDimension(layers.Layer):
    """Relative attention bias in nd factorizes over dimensions."""

    def __init__(self, lengths, number_of_heads, **kwargs):
        self.number_of_heads = number_of_heads
        self.lengths = lengths
        super(RelativeAttentionBiasNumberDimension, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.biases = []
        self.total_length = 1
        for i, lenth_ in enumerate(self.lengths):
            self.total_length *= lenth_
            if lenth_ > 1:
                weight = self.add_weight(
                    name='relative_attention_bias_%d' % i,
                    shape=[self.number_of_heads, 2 * lenth_],
                    initializer=tf.keras.initializers.Zeros(), trainable=True)
            else:
                weight = None
            self.biases.append(weight)

        super(RelativeAttentionBiasNumberDimension, self).build(input_shapes)

    def call(self, inputs=None):
        tile, index, biases = 1, None, []
        length_query = self.total_length

        for i, length_2 in enumerate(self.lengths):
            # Relative attention in every dimension separately.
            if length_2 > 1:
                new_bias = att_utils.relative_attention_bias(
                    self.biases[i], self.number_of_heads, index)
                repeat = self.total_length // (tile * length_2)
                if repeat > 1:
                    new_bias = tf.expand_dims(new_bias, -1)
                    new_bias = tf.tile(new_bias, [tile, repeat, tile, repeat])
                    new_bias = tf.reshape(new_bias,
                                          [length_query, self.number_of_heads,
                                           self.total_length])
                elif tile > 1:
                    new_bias = tf.tile(new_bias, [tile, 1, tile])
                tile *= length_2
                biases.append(new_bias)

        return tf.add_n(biases)


class ConditionalLayerNormalization(layers.Layer):
    """Conditional Layer Norm.

    Normalization of the input with the scale and shift as a function of 3-D
    context. Transforms 3-D spatial context into 1-D shift and scale of the
    layer-norm parameters. This is done via two dense projections:
      1. Spatial averaging via spatial_average='mean' or 'learnable'.
      2. Pointwise dense projection across channels.
    """

    def __init__(self,
                 spatial_average='learnable',
                 sequence='sc',
                 out_initial='glorot_uniform',
                 out_act='identity', **kwargs):
        super(ConditionalLayerNormalization, self).__init__(**kwargs)
        self.spatial_average = spatial_average
        self.sequence = sequence
        self.out_initial = out_initial
        self.out_act = out_act
        self.out_act_function = base_utils.act_to_function(out_act)
        if self.spatial_average not in ['mean', 'learnable']:
            raise ValueError(
                'Expected spatial average to be "mean" or "learnable" ,'
                'got %s' % self.spatial_average)
        if self.sequence not in ['sc', 'cs']:
            raise ValueError('Expected sequence to be "sc" or "cs" ,'
                             'got %s' % self.sequence)

    def build(self, input_shape):
        input_shape_0 = input_shape[0]
        height, width, features = input_shape_0[-3:]
        self.layer_normalization = layers.LayerNormalization(
            trainable=False, name='normalize')

        if self.spatial_average == 'learnable':
            self.spatial_weights = self.add_weight(
                name='spatial_average', shape=(1, height, width, 1),
                initializer=tf.keras.initializers.Ones())
        self.channel_dense = layers.Dense(
            units=2 * features, kernel_initializer=self.out_initial)
        super(ConditionalLayerNormalization, self).build(input_shape)

    def spatial_projection(self, conditional_inputs):
        if self.spatial_average == 'learnable':
            conditional_inputs = self.spatial_weights * conditional_inputs
        return tf.reduce_mean(conditional_inputs, axis=(1, 2), keepdims=True)

    def call(self, inputs):
        inputs, conditional_inputs = inputs

        if self.sequence == 'sc':
            operators = [self.spatial_projection, self.channel_dense]
        elif self.sequence == 'cs':
            operators = [self.channel_dense, self.spatial_projection]

        for operator_ in operators:
            conditional_inputs = operator_(conditional_inputs)

        scale, shift = tf.split(conditional_inputs, num_or_size_splits=2,
                                axis=-1)
        scale = self.out_act_function(scale)
        shift = self.out_act_function(shift)
        inputs_normalization = self.layer_normalization(inputs)
        inputs_normalization *= scale
        inputs_normalization += shift
        return inputs_normalization


class SelfAttentionNumberDimension(layers.Layer):
    """Transforms input through a N-D self-attention layer.

    Assume key, query and memory tensors are N-D tensors.

    1. Project key, query and value tensors into (N+2)-D tensors using
       dense layers where the outer two dimensions are
       [num_heads, num_channels_per_head].
       num_channels_per_head is set to num_channels // num_heads by default.
    2. Computes self-attention tensor using 2 dot products.
       The first computes similarity between the key and query tensors.
       The second uses this similarity to perform a weighted average over
       the value tensors. Done in _dot_product and _weighted_sum.
    3. The default behaviour, i.e if nd_block is not set, is to do global
       self attention. If nd_block_set is set, the above self-attention is limited
       to a block-size of nd_block_size.
       For instance, in case of 2D inputs (images), setting nd_block_size to
       [1, num_columns] or [num_rows, 1] to limit attention to column
       and rows respectively.
    4. If mask=='future', zero out the contribution of the values that
       violate raster ordering. Done in _apply_mask_and_bias
       for more details.
    5. Project the transformed tensor into hidden_size number of channels
       using a dense layer.

    Self-attention can be optionally conditioned with an tuple of two values
    where the second argument is the conditional input. Supports:
    1. Biasing: By setting cond_q, cond_k or cond_v to be True.
    2. Scaling: By setting cond_scale to be True.
    """

    def __init__(self,
                 hidden_size,
                 number_of_heads=1,
                 number_of_channels_per_head=None,
                 mask=None,
                 kernel_initializer='glorot_uniform',
                 number_dimension_block_size=None,
                 resolution=None,
                 conditional_initial='glorot_uniform',
                 conditional_key=False,
                 conditional_query=False,
                 conditional_value=False,
                 conditional_scale=False,
                 conditonal_act='identity',
                 **kwargs):
        super(SelfAttentionNumberDimension, self).__init__(**kwargs)
        if number_dimension_block_size:
            number_dimension_block_size = list(number_dimension_block_size)
        number_of_channels_per_head = number_of_channels_per_head or hidden_size // number_of_heads
        self.number_of_filters = [number_of_heads, number_of_channels_per_head]
        self.kernel_initializer = kernel_initializer
        self.hidden_size = hidden_size
        self.conditional_key = conditional_key
        self.conditional_query = conditional_query
        self.conditional_value = conditional_value
        self.conditional_scale = conditional_scale
        self.conditional_initial = conditional_initial
        self.conditional_act_function = base_utils.act_to_function(
            conditonal_act)
        self.project_conditional_query, self.project_conditional_key, self.project_conditional_value = None, None, None
        self.condition_filters = self.number_of_filters
        if conditional_scale:
            self.condition_filters = [number_of_heads,
                                      2 * number_of_channels_per_head]

        self.number_dimension_block_size = number_dimension_block_size
        self.resolution = resolution
        self.mask = mask
        self.number_of_channels_per_head = number_of_channels_per_head
        self.number_of_heads = number_of_heads
        self.hidden_size = hidden_size

        # By default, apply attention in third last dimension.
        # Last 2 dimensions are heads, channels.
        self.attention_dimension_query = self.attention_dimension_key = -3

        # Self attention type.
        self.is_block_attention = True if self.number_dimension_block_size else False

    def get_num_filters(self, is_conditional):
        if not is_conditional:
            return self.number_of_filters
        number_of_heads, number_of_channels_per_head = self.number_of_filters
        return [number_of_heads, 2 * number_of_channels_per_head]

    def condition_shift_and_scale(self, inputs, condition_inputs, is_conditional,
                                  layer):
        if not is_conditional:
            return inputs
        condition_out = layer(condition_inputs)
        if self.conditional_scale:
            scale, shift = tf.split(condition_out, num_or_size_splits=2,
                                    axis=-1)
            scale = self.conditional_act_function(scale)
            shift = self.conditional_act_function(shift)
            inputs *= scale
            inputs += shift
        else:
            inputs += condition_out
        return inputs

    def build(self, input_shape):
        if not isinstance(input_shape[-1], int):
            input_shape = input_shape[0]
        lengths = self.number_dimension_block_size or self.resolution or input_shape[
                                                                         1:-1]

        self.project_query = DenseNumberDimension(
            self.number_of_filters, kernel_initializer=self.kernel_initializer,
            name='q')
        self.project_key = DenseNumberDimension(
            self.number_of_filters, kernel_initializer=self.kernel_initializer,
            name='k')
        self.project_value = DenseNumberDimension(
            self.number_of_filters, kernel_initializer=self.kernel_initializer,
            name='v')
        self.project_final = DenseNumberDimension(
            self.hidden_size, kernel_initializer=self.kernel_initializer,
            contract_axes=2, name='output')

        self.relative_attention = RelativeAttentionBiasNumberDimension(
            lengths, self.number_of_heads)
        self.relative_attention.build([])

        if self.conditional_key:
            self.project_conditional_key = DenseNumberDimension(
                self.condition_filters,
                kernel_initializer=self.conditional_initial, name='cond_k')
        if self.conditional_query:
            self.project_conditional_query = DenseNumberDimension(
                self.condition_filters,
                kernel_initializer=self.conditional_initial, name='cond_q')
        if self.conditional_value:
            self.project_conditional_value = DenseNumberDimension(
                self.condition_filters,
                kernel_initializer=self.conditional_initial, name='cond_v')

        self.is_one_dimension_attention = (
                self.is_block_attention and
                sum(s != 1 for s in self.number_dimension_block_size) == 1)
        if self.is_one_dimension_attention:
            max_dimension = self.number_dimension_block_size.index(
                max(self.number_dimension_block_size))
            if self.number_dimension_block_size[max_dimension] == lengths[
                max_dimension]:
                self.is_block_attention = False
                self.attention_dimension_query = max_dimension - len(
                    self.number_dimension_block_size) - 2
                self.attention_dimension_key = self.attention_dimension_query
            else:
                self.is_one_dimension_attention = False

        if self.mask:
            total_length = functools.reduce(operator.mul, lengths, 1)
            self._mask = np.triu(
                np.ones([total_length, total_length], np.float32))
            if self.mask != Masking.FUTURE_PRESENT:
                self._mask *= (1.0 - np.eye(total_length))
            self._mask *= -1e6
            self._mask = tf.constant(
                np.reshape(self._mask, [total_length, 1, total_length]))

        super(SelfAttentionNumberDimension, self).build(input_shape)

    def _apply_mask_and_bias(self, alphas):
        bias = self.relative_attention(None)
        if self.mask:
            bias += self._mask

        expand_bias_dimensions = -self.attention_dimension_query - 3
        if expand_bias_dimensions:
            bias = tf.reshape(bias, [-1] + [1] * expand_bias_dimensions +
                              list(bias.shape[1:]))
        return alphas + bias

    def _dot_product(self, query, key, contract_dimension_query=-3,
                     contract_dimension_key=-3):
        number_of_batch_axes = len(query.shape) + contract_dimension_query
        pre_str = 'abcdefghij'[:number_of_batch_axes]
        in_dimension_q = -contract_dimension_query - 2
        in_dimension_k = -contract_dimension_key - 2

        in_str_query = 'zyxwv'[:in_dimension_q]
        in_str_key = 'zyxwv'[:in_dimension_k]
        einsum_str = '{}Q{}C,{}M{}C->{}Q{}M'.format(pre_str, in_str_query,
                                                    pre_str,
                                                    in_str_key, pre_str,
                                                    in_str_query)
        return tf.einsum(einsum_str, query, key)

    def _weighted_sum(self, alphas, value, contract_dimension_alpha=-3,
                      contract_dimension_value=-3):
        number_of_batch_axes = len(alphas.shape) + contract_dimension_alpha
        pre_str = 'abcdefghij'[:number_of_batch_axes]
        in_dimension_alpha = -contract_dimension_alpha - 2
        in_dimension_value = -contract_dimension_value - 2
        in_str_alpha = 'zyxwv'[:in_dimension_alpha]
        in_str_value = 'zyxwv'[:in_dimension_value]
        einsum_str = '{}Q{}M,{}M{}C->{}Q{}C'.format(pre_str, in_str_alpha,
                                                    pre_str,
                                                    in_str_value, pre_str,
                                                    in_str_alpha)
        return tf.einsum(einsum_str, alphas, value)

    def _prepare_block_attention(self, copy_inputs):
        return att_utils.divide_ndarray_blocks(copy_inputs,
                                               self.number_dimension_block_size,
                                               collapse=True)

    def _prepare_full_attention(self, copy_inputs):
        return tf.reshape(copy_inputs,
                          [copy_inputs.shape[0], -1, copy_inputs.shape[-1]])

    def call(self, inputs):
        condition_inputs = memory = None
        condition_qkv = self.conditional_value or self.conditional_query or self.conditional_key
        if condition_qkv:
            if tf.is_tensor(inputs) or len(inputs) != 2:
                raise ValueError('Expected tuple of (inputs, condition_inputs)')
            inputs, condition_inputs = inputs

        copy_inputs = inputs
        if not self.is_one_dimension_attention:
            # We flatten the index axes here. [B, ..., D] --> [B, M, D].
            if self.is_block_attention:
                copy_inputs = self._prepare_block_attention(copy_inputs)
            else:
                copy_inputs = self._prepare_full_attention(copy_inputs)
        memory = copy_inputs
        query, key, value = self.project_query(copy_inputs), self.project_key(
            memory), self.project_value(memory)

        query = self.condition_shift_and_scale(
            query, condition_inputs, self.conditional_query,
            self.project_conditional_query)
        key = self.condition_shift_and_scale(
            key, condition_inputs, self.conditional_key,
            self.project_conditional_key)
        value = self.condition_shift_and_scale(
            value, condition_inputs, self.conditional_value,
            self.project_conditional_value)

        query *= query.shape[-1] ** -0.5
        alphas = self._dot_product(query, key, self.attention_dimension_query,
                                   self.attention_dimension_key)
        alphas = self._apply_mask_and_bias(alphas)
        weights = tf.nn.softmax(alphas)
        output = self._weighted_sum(weights, value,
                                    self.attention_dimension_query,
                                    self.attention_dimension_key)
        output = self.project_final(output)
        return output


class FactorizedAttention(layers.Layer):
    """Encodes image into 2-D spatial context with factorized attention layers."""

    def __init__(self, config, **kwargs):
        super(FactorizedAttention, self).__init__(**kwargs)
        self.config = config
        self.dropout = self.config.get('dropout', 0.0)

    def build(self, input_shapes):
        first_frame_size, hidden_size = self.config.first_frame_size, self.config.hidden_size
        number_of_heads = self.config.number_of_heads
        height, width = input_shapes[1:3]

        self.position_embed = PositionEmbed(axes=[1, 2],
                                            max_lengths=[height, width])

        self.residual_layers = []
        number_of_normalizations = 4 * self.config.number_of_encoder_layers
        self.layer_normalizations = [layers.LayerNormalization() for _ in
                                     range(number_of_normalizations)]

        for _ in range(self.config.number_of_encoder_layers):
            # unmasked row
            unmask_row = SelfAttentionNumberDimension(
                hidden_size=hidden_size, number_of_heads=number_of_heads,
                number_dimension_block_size=[1, width],
                resolution=[height, width])

            first_frame_row = tf.keras.Sequential([
                layers.Dense(units=first_frame_size, activation='relu'),
                layers.Dense(units=hidden_size)
            ])

            # unmasked column,
            unmask_column = SelfAttentionNumberDimension(
                hidden_size=hidden_size, number_of_heads=number_of_heads,
                number_dimension_block_size=[height, 1],
                resolution=[height, width])

            first_frame_column = tf.keras.Sequential([
                layers.Dense(units=first_frame_size, activation='relu'),
                layers.Dense(units=hidden_size)
            ])

            self.residual_layers.append(unmask_row)
            self.residual_layers.append(first_frame_row)
            self.residual_layers.append(unmask_column)
            self.residual_layers.append(first_frame_column)

    def call(self, inputs, training=True):
        inputs = self.position_embed(inputs)

        # Apply a stack of unmaked row and column attention layers.
        for layer, normalization in zip(self.residual_layers,
                                        self.layer_normalizations):
            output = layer(inputs)
            output = residual_dropout(inputs, output, self.dropout, training)
            inputs = normalization(output)

        return inputs
