import torch
import torch.nn.functional as torch_functional
import torch.nn as torch_neural_network
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal, normal_style
from function import calculate_mean_standard
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, truncated_normal_
from torchvision.utils import save_image


class PatchEmbed(torch_neural_network.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, image_size=256, patch_size=8, in_chanals=3, embed_dimension=512):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        number_of_patches = (image_size[1] // patch_size[1]) * (
                image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.number_of_patches = number_of_patches

        self.project = torch_neural_network.Conv2d(in_chanals, embed_dimension, kernel_size=patch_size,
                                                   stride=patch_size)
        self.upsample1 = torch_neural_network.Upsample(scale_factor=2, mode='nearest')

    def forward(self, patch_embedding):
        batch, channal, height, width = patch_embedding.shape
        patch_embedding = self.project(patch_embedding)

        return patch_embedding


decoder = torch_neural_network.Sequential(
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 256, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.Upsample(scale_factor=2, mode='nearest'),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 256, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 256, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 256, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 128, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.Upsample(scale_factor=2, mode='nearest'),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(128, 128, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(128, 64, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.Upsample(scale_factor=2, mode='nearest'),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(64, 64, (3, 3)),
    torch_neural_network.ReLU(),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(64, 3, (3, 3)),
)

vgg = torch_neural_network.Sequential(
    torch_neural_network.Conv2d(3, 3, (1, 1)),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(3, 64, (3, 3)),
    torch_neural_network.ReLU(),  # relu1-1
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(64, 64, (3, 3)),
    torch_neural_network.ReLU(),  # relu1-2
    torch_neural_network.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(64, 128, (3, 3)),
    torch_neural_network.ReLU(),  # relu2-1
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(128, 128, (3, 3)),
    torch_neural_network.ReLU(),  # relu2-2
    torch_neural_network.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(128, 256, (3, 3)),
    torch_neural_network.ReLU(),  # relu3-1
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 256, (3, 3)),
    torch_neural_network.ReLU(),  # relu3-2
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 256, (3, 3)),
    torch_neural_network.ReLU(),  # relu3-3
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 256, (3, 3)),
    torch_neural_network.ReLU(),  # relu3-4
    torch_neural_network.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(256, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu4-1, this is the last layer used
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu4-2
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu4-3
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu4-4
    torch_neural_network.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu5-1
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu5-2
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU(),  # relu5-3
    torch_neural_network.ReflectionPad2d((1, 1, 1, 1)),
    torch_neural_network.Conv2d(512, 512, (3, 3)),
    torch_neural_network.ReLU()  # relu5-4
)


class MultiLayerPerceptron(torch_neural_network.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dimension, hidden_dimension, output_dimension, number_of_layers):
        super().__init__()
        self.number_of_layers = number_of_layers
        hidden = [hidden_dimension] * (number_of_layers - 1)
        self.layers = torch_neural_network.ModuleList(
            torch_neural_network.Linear(n, k) for n, k in zip([input_dimension] + hidden, hidden + [output_dimension]))

    def forward(self, multilayer_perceptron_neural):
        for i, layer in enumerate(self.layers):
            multilayer_perceptron_neural = torch_functional.relu(layer(multilayer_perceptron_neural)) if i < self.number_of_layers - 1 else layer(multilayer_perceptron_neural)
        return multilayer_perceptron_neural


class StyTrans(torch_neural_network.Module):
    """ This is the style transform transformer module """

    def __init__(self, encoder, decoder, PatchEmbed, transformer, arguments, evaluation=False):

        super().__init__()
        encoder_layers = list(encoder.children())
        self.encoder_1 = torch_neural_network.Sequential(*encoder_layers[:4])  # input -> relu1_1
        self.encoder_2 = torch_neural_network.Sequential(*encoder_layers[4:11])  # relu1_1 -> relu2_1
        self.encoder_3 = torch_neural_network.Sequential(*encoder_layers[11:18])  # relu2_1 -> relu3_1
        self.encoder_4 = torch_neural_network.Sequential(*encoder_layers[18:31])  # relu3_1 -> relu4_1
        self.encoder_5 = torch_neural_network.Sequential(*encoder_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['encoder_1', 'encoder_2', 'encoder_3', 'encoder_4', 'encoder_5']:
            for parameter_ in getattr(self, name).parameters():
                parameter_.requires_grad = False

        self.mse_loss = torch_neural_network.MSELoss()
        self.transformer = transformer
        hidden_dimension = transformer.dimension_model
        self.decode = decoder
        self.embedding = PatchEmbed
        self.evaluation = evaluation

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            function = getattr(self, 'encoder_{:d}'.format(i + 1))
            results.append(function(results[-1]))
        return results[1:]

    def calculate_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calculate_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_standard = calculate_mean_standard(input)
        target_mean, target_standard = calculate_mean_standard(target)
        return (self.mse_loss(input_mean, target_mean) +
                self.mse_loss(input_standard, target_standard))

    def forward(self, samples_content: NestedTensor, samples_style: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        content_input = samples_content
        style_input = samples_style
        if isinstance(samples_content, (list, torch.Tensor)):
            samples_content = nested_tensor_from_tensor_list(
                samples_content)  # support different-sized images padding is used for mask [tensor, mask]
        if isinstance(samples_style, (list, torch.Tensor)):
            samples_style = nested_tensor_from_tensor_list(samples_style)

            # ### features used to calcate loss
        content_features = self.encode_with_intermediate(samples_content.tensors)
        style_features = self.encode_with_intermediate(samples_style.tensors)

        ### Linear projection
        style = self.embedding(samples_style.tensors)
        content = self.embedding(samples_content.tensors)

        # postional embedding is calculated in transformer.py
        position_style = None
        position_content = None

        mask = None
        transformer_output_neural = self.transformer(style, mask, content, position_content, position_style)
        output_image = self.decode(transformer_output_neural)

        ##########################################################################
        if not self.evaluation:
            output_image_features = self.encode_with_intermediate(output_image)
            loss_content = self.calculate_content_loss(normal(output_image_features[-1]), normal(
                content_features[-1])) + self.calculate_content_loss(normal(output_image_features[-2]),
                                                                  normal(
                                                                 content_features[-2]))
            # Style loss
            loss_style = self.calculate_style_loss(output_image_features[0], style_features[0])
            for i in range(1, 5):
                loss_style += self.calculate_style_loss(output_image_features[i], style_features[i])

            decoded_content = self.decode(
                self.transformer(content, mask, content, position_content, position_content))
            decoded_style = self.decode(self.transformer(style, mask, style, position_style, position_style))

            # Identity losses lambda 1
            loss_lambda_style = self.calculate_content_loss(decoded_content,
                                                       content_input) + self.calculate_content_loss(
                decoded_style, style_input)

            # Identity losses lambda 2
            decoded_content_features = self.encode_with_intermediate(decoded_content)
            decoded_style_features = self.encode_with_intermediate(decoded_style)
            loss_lambda_content = self.calculate_content_loss(decoded_content_features[0], content_features[
                0]) + self.calculate_content_loss(decoded_style_features[0], style_features[0])
            for i in range(1, 5):
                loss_lambda_content += self.calculate_content_loss(decoded_content_features[i], content_features[
                    i]) + self.calculate_content_loss(decoded_style_features[i], style_features[i])
            # Please select and comment out one of the following two sentences
            return output_image, loss_content, loss_style, loss_lambda_style, loss_lambda_content  # train

        elif self.evaluation:
            return output_image    #test
