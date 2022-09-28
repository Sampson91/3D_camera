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

"""Some utils for attention layers."""

import functools
import itertools
import operator
import numpy as np
import tensorflow.compat.v2 as tf


def index_to_step(index, shape):
    """Compute step for a given nd index if we were enumerating to shape."""
    step = index[0]
    for i, step_ in enumerate(shape[1:]):
        step = step * step_ + index[i + 1]
    return step


def pad_to_multiple_ndarray(tensor, shape):
    """Pads x such that nd-shape axes are multiples of shape axes.

    Args:
      tensor: Tensor of shape [B] + nd_shape + [...].
      shape: Shape tuple of same length as nd_shape.

    Returns:
      x padded to make each axis in nd_shape divisible by the same shape axis.
    """
    # x == tensor
    tensor_shape = tensor.shape.as_list()
    number_of_feature_dimension = len(tensor_shape) - len(shape) - 1
    if all(shape_ for shape_ in tensor_shape[1:len(shape) + 1]):
        pad_amount = np.mod(-np.asarray(tensor_shape[1:len(shape) + 1]), shape)
        paddings = [[0, 0]] + [[0, pad_] for pad_ in pad_amount] + [
            [0, 0]] * number_of_feature_dimension

        return tf.pad(tensor, paddings) if any(
            any(padding_) for padding_ in paddings) else tensor
    else:
        # If shape is not fully defined.
        tensorflow_shape = tf.shape(tensor)
        last = tensor_shape[-number_of_feature_dimension:]
        paddings = [
            [0, -(tensor_shape[i + 1] or tensorflow_shape[i + 1]) % shape_]
            for i, shape_ in enumerate(shape)]
        paddings = [[0, 0]] + paddings + [[0, 0]] * number_of_feature_dimension
        padded_tensor = tf.pad(tensor, paddings)
        padded_shape = padded_tensor.shape.as_list()
        padded_shape = padded_shape[:-1] + last
        return padded_tensor


def divide_ndarray_blocks(inputs, ndarray_block_size, collapse=False):
    """Divides input into non-overlapping n-dimensional blocks.

    Args:
      inputs: [B, D1, D2, ..., Dk, ...] tensor.
      ndarray_block_size: Shape tuple of length k.
      collapse: collapse.

    Returns:
      A [B, D1 // S1, D2 // S2, ..., Dk // Sk, S1 , S2 , ... , Sk, ...] tensor.
    """
    ndarray_block_size = list(ndarray_block_size)
    inputs = pad_to_multiple_ndarray(inputs, ndarray_block_size)

    shape = list(inputs.shape)
    for i, shape_ in enumerate(shape):
        if shape_ is None:
            shape[i] = tf.shape(inputs)[i]

    block_axes = shape[1:len(ndarray_block_size) + 1]
    number_of_blocks = [full_size_ // size_ for full_size_, size_ in
                        zip(block_axes, ndarray_block_size)]
    number_of_ndarray_axes = len(ndarray_block_size)
    number_of_feature_axes = len(shape) - number_of_ndarray_axes - 1
    features_shape = shape[-number_of_feature_axes:]

    # Reshape into [B, D1 // S1, S1, D2 // S2, S2, ..., Dk // Sk, Sk, ...].
    middle_shape = list(
        itertools.chain(*zip(number_of_blocks, ndarray_block_size)))
    cut_shape = shape[:1] + middle_shape + features_shape
    cut_inputs = tf.reshape(inputs, cut_shape)

    # Permute into [B, D1 // S1, D2 // S2, ..., Dk // Sk, S1, S2, ..., Sk, ...].
    number_of_middle_axes = number_of_ndarray_axes * 2
    number_of_feature_axes = len(shape) - number_of_ndarray_axes - 1
    middle_permute = itertools.chain(
        range(1, number_of_middle_axes, 2),
        range(2, number_of_middle_axes + 1, 2))
    post_permute = range(number_of_middle_axes + 1,
                         number_of_middle_axes + number_of_feature_axes + 1)
    permutation = [0] + list(middle_permute) + list(post_permute)
    permuted_inputs = tf.transpose(cut_inputs, permutation)

    if not collapse:
        return permuted_inputs
    # Collapse to [B * D1 // S1 * D2 // S2 * ... * Dk // Sk, S1 * S2 * Sk, ...]
    block_length = functools.reduce(operator.mul, ndarray_block_size, 1)
    collapsed_inputs = tf.reshape(permuted_inputs, [-1, block_length] +
                                  features_shape)

    return collapsed_inputs


def relative_attention_bias(relative_bias, number_of_heads, decode_step=None):
    """Computes attention bias based on relative positions.

    Content-based relative position attention bias was used in:
      https://arxiv.org/pdf/1803.02155.
    Non-content-based relative position attention bias was used in:
      https://arxiv.org/abs/1606.01933.

    Args:
      relative_bias: Relative bias variable of shape [num_heads, 2 * length].
      number_of_heads: Number of attention heads.
      decode_step: Optional decode step, used for slicing during decoding.

    Returns:
      A [..., length, num_heads, length] tensor with queries.
    """
    number_of_relative_positions = relative_bias.shape[-1]
    length = number_of_relative_positions // 2

    if tf.is_tensor(decode_step):
        # This is decoding so we need to select the current slice within rel_bias.
        # E.g.: len_k = 3, decode_step = 1
        # We have: rel_bias = [-2, -1, 0, 1, 2, 3]
        # We want: [-1, 0, 1]
        # We slice at len_k - decode_step - 1 = 1
        relative_bias = tf.reshape(relative_bias, [1, number_of_heads,
                                                   number_of_relative_positions])
        start = ((length - 1) - decode_step)
        relative_bias = tf.slice(relative_bias, [0, 0, start],
                                 [1, number_of_heads, length])
        return relative_bias

    # Now we have to shift in order to compute relative biases.
    # Example: length = 3
    # Say we want:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Start: [[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]]
    # We linearize: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3]
    # We slice: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0]
    # We reshape: [[-2, -1, 0, 1, 2], [3, -2, -1, 0, 1], [2, 3, -2, -1, 0]]
    # We slice: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Tadaaa!

    # [heads, len_q * number_of_relative_positions]
    relative_bias = tf.tile(relative_bias, [1, length])

    # [heads, len_q * (number_of_relative_positions - 1)]
    number_of_relative_positions -= 1
    relative_bias = relative_bias[Ellipsis,
                    :length * number_of_relative_positions]

    # [heads, len_q, number_of_relative_positions - 1]
    # Now every row is shifted by 1 to the right.
    relative_bias = tf.reshape(relative_bias, [number_of_heads, length,
                                               number_of_relative_positions])

    # [heads, len_q, len_k]
    # Slice the overlapping elements from start.
    relative_bias = relative_bias[Ellipsis,
                    number_of_relative_positions - length:]
    # [len_q, heads, len_k]
    relative_bias = tf.transpose(relative_bias, [1, 0, 2])

    return relative_bias
