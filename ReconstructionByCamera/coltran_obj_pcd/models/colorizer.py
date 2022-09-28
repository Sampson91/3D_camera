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

"""ColTran core.

Core autoregressive component of the colorization transformer based on
the AxialTransformer with conditional self-attention layers.

See Section 3 and Section 4.1 of https://openreview.net/pdf?id=5NA1PinlGFu
for more details.
"""
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from models import core
from models import layers as coltran_layers
from utils import base_utils


class ColTranCore(tf.keras.Model):
    """Colorization Transformer."""

    def __init__(self, config, **kwargs):
        super(ColTranCore, self).__init__(**kwargs)
        self.config = config

        # 3 bits per channel, 8 colors per channel, a total of 512 colors.
        self.number_of_symbols_per_channel = 2 ** 3
        self.number_of_symbols = self.number_of_symbols_per_channel ** 3
        self.gray_symbols, self.number_of_channels = 256, 1

        self.encoder_config = config.encoder
        self.decoder_config = config.decoder
        self.hidden_size = self.config.get('hidden_size',
                                           self.decoder_config.hidden_size)

        # stage can be 'encoder_decoder' or 'decoder'
        # 1. decoder -> loss only due to autoregressive model.
        # 2. encoder_decoder -> loss due to both the autoregressive and parallel
        # model.
        # encoder_only and all
        self.stage = config.get('stage', 'decoder')
        self.is_parallel_loss = 'encoder' in self.stage
        stages = ['decoder', 'encoder_decoder']
        if self.stage not in stages:
            raise ValueError('Expected stage to be in %s, got %s' %
                             (str(stages), self.stage))

    @property
    def metric_keys(self):
        if self.stage == 'encoder_decoder':
            return ['encoder']
        return []

    def build(self, input_shape):
        # encoder graph
        self.encoder = core.GrayScaleEncoder(self.encoder_config)
        if self.is_parallel_loss:
            self.parallel_dense = layers.Dense(
                units=self.number_of_symbols, name='parallel_logits',
                use_bias=False)

        # decoder graph: outer decoder -> inner decoder -> logits.
        self.pixel_embed_layer = layers.Dense(
            units=self.hidden_size, use_bias=False)
        self.outer_decoder = core.OuterDecoder(self.decoder_config)
        self.inner_decoder = core.InnerDecoder(self.decoder_config)
        self.final_dense = layers.Dense(
            units=self.number_of_symbols, name='auto_logits')
        self.final_normalization = layers.LayerNormalization()

    def call(self, inputs, training=True):
        # encodes grayscale (H, W) into activations of shape (H, W, 512).
        gray = tf.image.rgb_to_grayscale(inputs)
        encoded_gray = self.encoder(gray)

        if self.is_parallel_loss:
            encoder_logits = self.parallel_dense(encoded_gray)
            encoder_logits = tf.expand_dims(encoder_logits, axis=-2)

        decoder_logits = self.decoder(inputs, encoded_gray, training=training)
        # decoder_logits 可以打印
        if self.is_parallel_loss:
            return decoder_logits, {'encoder_logits': encoder_logits}
        return decoder_logits, {}

    def decoder(self, inputs, gray_scale, training):
        """Decodes grayscale representation and masked colors into logits."""
        # (H, W, 512) preprocessing.
        # quantize to 3 bits.
        labels = base_utils.convert_bits(inputs, number_of_bits_in=8,
                                         number_of_bits_out=3)

        # bin each channel triplet -> (H, W, 3) with 8 possible symbols
        # (H, W, 512)
        labels = base_utils.labels_to_bins(labels,
                                           self.number_of_symbols_per_channel)

        # (H, W) with 512 symbols to (H, W, 512)
        labels = tf.one_hot(labels, depth=self.number_of_symbols)

        hot_decoder = self.pixel_embed_layer(labels)
        hot_upper = self.outer_decoder((hot_decoder, gray_scale),
                                       training=training)
        hot_inner = self.inner_decoder((hot_decoder, hot_upper, gray_scale),
                                       training=training)

        activations = self.final_normalization(hot_inner)
        logits = self.final_dense(activations)
        # 可以打印 logits
        return tf.expand_dims(logits, axis=-2)

    def image_loss(self, logits, labels):
        """Cross-entropy between the logits and labels."""
        height, width = labels.shape[1:3]
        logits = tf.squeeze(logits, axis=-2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(loss, axis=0)
        loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
        return loss / (height * width)

    def loss(self, targets, logits, train_config, training,
             auxiliary_output=None):
        """Converts targets to coarse colors and computes log-likelihood."""
        downsample = train_config.get('downsample', False)
        downsample_res = train_config.get('downsample_res', 64)
        if downsample:
            labels = targets['targets_%d' % downsample_res]
        else:
            labels = targets['targets']

        if auxiliary_output is None:
            auxiliary_output = {}

        # quantize labels.
        labels = base_utils.convert_bits(labels, number_of_bits_in=8,
                                         number_of_bits_out=3)

        # bin each channel triplet.
        labels = base_utils.labels_to_bins(labels,
                                           self.number_of_symbols_per_channel)

        loss = self.image_loss(logits, labels)
        encoder_logits = auxiliary_output.get('encoder_logits')
        if encoder_logits is None:
            return loss, {}

        encoder_loss = self.image_loss(encoder_logits, labels)
        return loss, {'encoder': encoder_loss}

    def get_logits(self, inputs_dictionary, train_config, training):
        is_downsample = train_config.get('downsample', False)
        downsample_resolution = train_config.get('downsample_resolution', 64)
        if is_downsample:
            inputs = inputs_dictionary['targets_%d' % downsample_resolution]
        else:
            inputs = inputs_dictionary['targets']
        return self(inputs=inputs, training=training)

    def sample(self, gray_condition, mode='argmax'):
        output = {}

        zipped_gray = self.encoder(gray_condition, training=False)
        if self.is_parallel_loss:
            zipped_logits = self.parallel_dense(zipped_gray)
            parallel_image = tf.argmax(zipped_logits, axis=-1,
                                       output_type=tf.int32)
            parallel_image = self.post_process_image(parallel_image)

            output['parallel'] = parallel_image

        image, probability = self.autoregressive_sample(zipped_gray=zipped_gray,
                                                        mode=mode)
        output['auto_%s' % mode] = image
        output['probability'] = probability
        return output

    def autoregressive_sample(self, zipped_gray, mode='sample'):
        """Generates pixel-by-pixel.

        1. The encoder is run once per-channel.
        2. The outer decoder is run once per-row.
        3. the inner decoder is run once per-pixel.

        The context from the encoder and outer decoder conditions the
        inner decoder. The inner decoder then generates a row, one pixel at a time.

        After generating all pixels in a row, the outer decoder is run to recompute
        context. This condtions the inner decoder, which then generates the next
        row, pixel-by-pixel.

        Args:
          zipped_gray: grayscale image.
          mode: sample or argmax.

        Returns:
          image: coarse image of shape (B, H, W)
          image_probability: probalities, shape (B, H, W, 512)
        """
        number_of_filters = self.config.hidden_size
        batch_size, height, width = zipped_gray.shape[:3]

        # channel_cache[i, j] stores the pixel embedding for row i and colomn j.
        canvas_shape = (batch_size, height, width, number_of_filters)
        channel_cache = coltran_layers.Cache(canvas_shape=(height, width))
        initial_channel = tf.zeros(shape=canvas_shape)
        initial_index = tf.stack([0, 0])
        channel_cache(inputs=(initial_channel, initial_index))

        # upper_context[row_ind] stores context from all previously generated rows.
        upper_context = tf.zeros(shape=canvas_shape)

        # row_cache[0, j] stores the pixel embedding for the column j of the row
        # under generation. After every row is generated, this is rewritten.
        row_cache = coltran_layers.Cache(canvas_shape=(1, width))
        initial_row = tf.zeros(shape=(batch_size, 1, width, number_of_filters))
        row_cache(inputs=(initial_row, initial_index))

        pixel_samples, pixel_probabilities = [], []

        for row in range(height):
            row_condition_channel = tf.expand_dims(zipped_gray[:, row], axis=1)
            row_condition_upper = tf.expand_dims(upper_context[:, row], axis=1)
            row_cache.reset()

            generate_row, probability_row = [], []
            for colomn in range(width):
                inner_input = (
                    row_cache.cache, row_condition_upper, row_condition_channel)
                # computes output activations at colomn.
                activations = self.inner_decoder(inner_input, row_index=row,
                                                 training=False)

                pixel_sample, pixel_embed, pixel_probability = self.act_logit_sample_embed(
                    activations, colomn, mode=mode)
                probability_row.append(pixel_probability)
                generate_row.append(pixel_sample)

                # row_cache[:, colomn] = pixel_embed
                row_cache(inputs=(pixel_embed, tf.stack([0, colomn])))

                # channel_cache[row, colomn] = pixel_embed
                channel_cache(inputs=(pixel_embed, tf.stack([row, colomn])))

            generate_row = tf.stack(generate_row, axis=-1)
            pixel_samples.append(generate_row)
            pixel_probabilities.append(tf.stack(probability_row, axis=1))

            # after a row is generated, recomputes the context for the next row.
            # upper_context[row] = self_attention(channel_cache[:row_index])
            upper_context = self.outer_decoder(
                inputs=(channel_cache.cache, zipped_gray), training=False)

        image = tf.stack(pixel_samples, axis=1)
        image = self.post_process_image(image)

        image_probability = tf.stack(pixel_probabilities, axis=1)
        return image, image_probability

    def act_logit_sample_embed(self, activations, colomn_index, mode='sample'):
        """Converts activations[col_ind] to the output pixel.

        Activation -> Logit -> Sample -> Embedding.

        Args:
          activations: 5-D Tensor, shape=(batch_size, 1, width, hidden_size)
          colomn_index: integer.
          mode: 'sample' or 'argmax'
        Returns:
          pixel_sample: 1-D Tensor, shape=(batch_size, 1, 1)
          pixel_embed: 4-D Tensor, shape=(batch_size, 1, 1, hidden_size)
          pixel_probability: 4-D Tensor, shape=(batch_size, 1, 512)
        """
        batch_size = activations.shape[0]
        pixel_activation = tf.expand_dims(activations[:, :, colomn_index],
                                          axis=-2)
        pixel_logits = self.final_dense(
            self.final_normalization(pixel_activation))
        pixel_logits = tf.squeeze(pixel_logits, axis=[1, 2])
        pixel_probability = tf.nn.softmax(pixel_logits, axis=-1)

        if mode == 'sample':
            pixel_sample = tf.random.categorical(
                pixel_logits, num_samples=1, dtype=tf.int32)
            pixel_sample = tf.squeeze(pixel_sample, axis=-1)
        elif mode == 'argmax':
            pixel_sample = tf.argmax(pixel_logits, axis=-1,
                                     output_type=tf.int32)

        pixel_sample_expand = tf.reshape(pixel_sample, [batch_size, 1, 1])
        pixel_one_hot = tf.one_hot(pixel_sample_expand,
                                   depth=self.number_of_symbols)
        pixel_embed = self.pixel_embed_layer(pixel_one_hot)
        return pixel_sample, pixel_embed, pixel_probability

    def post_process_image(self, image):
        """Post process image of size (H, W, 512) to a coarse RGB image."""
        image = base_utils.bins_to_labels(
            image,
            number_of_symbols_per_channel=self.number_of_symbols_per_channel)
        image = base_utils.convert_bits(image, number_of_bits_in=3,
                                        number_of_bits_out=8)
        image = tf.cast(image, dtype=tf.uint8)
        return image
