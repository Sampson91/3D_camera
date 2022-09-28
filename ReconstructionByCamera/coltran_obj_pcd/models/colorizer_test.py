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

"""Tests for colorizer."""
from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
from models import colorizer


class ColTranCoreTest(tf.test.TestCase):

    def get_config(self, encoder_net='attention'):
        config = ConfigDict()
        config.image_bit_depth = 3
        config.encoder_1x1 = True
        config.resolution = [64, 64]
        config.batch_size = 2
        config.encoder_net = encoder_net
        config.hidden_size = 128
        config.stage = 'decoder'

        config.encoder = ConfigDict()
        config.encoder.dropout = 0.0
        config.encoder.first_frame_size = 128
        config.encoder.hidden_size = 128
        config.encoder.number_of_heads = 1
        config.encoder.number_of_encoder_layers = 1

        config.decoder = ConfigDict()
        config.decoder.first_frame_size = 128
        config.decoder.hidden_size = 128
        config.decoder.number_of_heads = 1
        config.decoder.number_of_outer_layers = 1
        config.decoder.number_of_inner_layers = 1
        config.decoder.resolution = [64, 64]
        config.decoder.dropout = 0.1
        config.decoder.condition_layer_normalization = True
        config.decoder.condition_query = True
        config.decoder.condition_key = True
        config.decoder.condition_value = True
        config.decoder.condition_query = True
        config.decoder.condition_scale = True
        config.decoder.conditional_multilayer_perceptron = 'affine'
        return config

    def test_transformer_attention_encoder(self):
        config = self.get_config(encoder_net='attention')
        config.stage = 'encoder_decoder'
        transformer = colorizer.ColTranCore(config=config)
        images = tf.random.uniform(shape=(2, 2, 2, 3), minval=0,
                                   maxval=256, dtype=tf.int32)
        logits = transformer(inputs=images, training=True)[0]
        self.assertEqual(logits.shape, (2, 2, 2, 1, 512))

        # batch-size=2
        gray = tf.image.rgb_to_grayscale(images)
        output = transformer.sample(gray, mode='argmax')
        output_numpy = output['auto_argmax'].numpy()
        probability_numpy = output['probability'].numpy()
        self.assertEqual(output_numpy.shape, (2, 2, 2, 3))
        self.assertEqual(probability_numpy.shape, (2, 2, 2, 512))
        # logging.info(output_numpy[0, ..., 0])

        # batch-size=1
        output_numpy_batch_size_1, probability_numpy_batch_size_1 = [], []
        for batch_index in [0, 1]:
            current_gray = tf.expand_dims(gray[batch_index], axis=0)
            current_out = transformer.sample(current_gray, mode='argmax')
            current_out_numpy = current_out['auto_argmax'].numpy()
            current_probability_numpy = current_out['probability'].numpy()
            output_numpy_batch_size_1.append(current_out_numpy)
            probability_numpy_batch_size_1.append(current_probability_numpy)
        output_numpy_batch_size_1 = np.concatenate(output_numpy_batch_size_1,
                                                   axis=0)
        probability_numpy_batch_size_1 = np.concatenate(
            probability_numpy_batch_size_1, axis=0)
        self.assertTrue(np.allclose(output_numpy, output_numpy_batch_size_1))
        self.assertTrue(
            np.allclose(probability_numpy, probability_numpy_batch_size_1))

    def test_transformer_encoder_decoder(self):
        config = self.get_config()
        config.stage = 'encoder_decoder'

        transformer = colorizer.ColTranCore(config=config)
        images = tf.random.uniform(shape=(1, 64, 64, 3), minval=0,
                                   maxval=256, dtype=tf.int32)
        logits, encoder_logits = transformer(inputs=images, training=True)

        encoder_logits = encoder_logits['encoder_logits']
        self.assertEqual(encoder_logits.shape, (1, 64, 64, 1, 512))
        self.assertEqual(logits.shape, (1, 64, 64, 1, 512))



if __name__ == '__main__':
    tf.test.main()
