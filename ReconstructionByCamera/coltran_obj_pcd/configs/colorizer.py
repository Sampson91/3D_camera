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

"""Test configurations for colorizer."""
from ml_collections import ConfigDict


def get_config():
    """Experiment configuration."""
    config = ConfigDict()

    # Data.
    config.dataset = 'imagenet'
    config.downsample = True
    #config.downsample = False
    config.downsample_res = 64
    config.resolution = [256, 256]

    # Training.
    config.batch_size = 3 #7
    config.max_train_steps = 10000 #450000
    config.save_checkpoint_secs = 60
    config.number_of_epochs = -1
    config.polyak_decay = 0.999
    config.evaluation_number_of_examples = 20000
    config.evaluation_batch_size = 16
    config.evaluation_checkpoint_wait_secs = -1

    # loss hparams.
    config.loss_factor = 0.99
    config.encoder_loss_factor = 0.01

    config.optimizer = ConfigDict()
    config.optimizer.type = 'rmsprop'
    config.optimizer.learning_rate = 3e-4

    # Model.
    config.model = ConfigDict()
    config.model.hidden_size = 512
    config.model.stage = 'encoder_decoder'
    config.model.resolution = [64, 64]
    config.model.name = 'coltran_core'

    # encoder
    config.model.encoder = ConfigDict()
    config.model.encoder.first_frame_size = 512
    config.model.encoder.hidden_size = 512
    config.model.encoder.number_of_heads = 4
    config.model.encoder.number_of_encoder_layers = 4
    config.model.encoder.dropout = 0.0

    # decoder
    config.model.decoder = ConfigDict()
    config.model.decoder.first_frame_size = 512
    config.model.decoder.hidden_size = 512
    config.model.decoder.resolution = [64, 64]
    config.model.decoder.number_of_heads = 4
    config.model.decoder.number_of_inner_layers = 2
    config.model.decoder.number_of_outer_layers = 2
    config.model.decoder.dropout = 0.0
    config.model.decoder.skip = True

    config.model.decoder.condition_multilayer_perceptron = 'affine'
    config.model.decoder.condition_multilayer_perceptron_act = 'identity'

    config.model.decoder.condition_layer_normalization_act = 'identity'
    config.model.decoder.condition_layer_normalization = True
    config.model.decoder.condition_layer_normalization_sequence = 'sc'
    config.model.decoder.condition_layer_normalization_spatial_pooling_average = 'learnable'
    config.model.decoder.condition_layer_normalization_initial = 'glorot_uniform'

    config.model.decoder.condition_attention_initial = 'glorot_uniform'
    config.model.decoder.condition_attention_value = True
    config.model.decoder.condition_attention_query = True
    config.model.decoder.condition_attention_key = True
    config.model.decoder.condition_attention_scale = True
    config.model.decoder.condition_attention_act = 'identity'

    config.sample = ConfigDict()
    config.sample.log_dir = ''
    config.sample.batch_size = 1
    config.sample.mode = 'sample'
    config.sample.number_of_samples = 1
    config.sample.number_of_outputs = 1
    config.sample.skip_batches = 0
    config.sample.generate_file = 'gen0'
    return config
