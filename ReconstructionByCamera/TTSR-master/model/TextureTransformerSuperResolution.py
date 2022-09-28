from model import MainNet, LearnableTextureExtractor, SearchTransfer

import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional


class TextureTransformerSuperResolution(torch_neural_network.Module):
    def __init__(self, args):
        super(TextureTransformerSuperResolution, self).__init__()
        self.args = args
        self.number_residual_blocks = list(
            map(int, args.number_residual_blocks.split('+')))
        self.MainNet = MainNet.MainNet(
            number_residual_blocks=self.number_residual_blocks,
            number_features=args.number_features,
            residual_scale=args.residual_scale)
        self.LearnableTextureExtractor = LearnableTextureExtractor.LearnableTextureExtractor(
            requires_grad=True)
        self.learnable_texture_extractor_copy = LearnableTextureExtractor.LearnableTextureExtractor(
            requires_grad=False)  ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()

    def forward(self, low_resolution_image=None,
                upsampled_low_resolution_image=None,
                high_resolution_images_as_references=None,
                down_and_upsampled_Ref_image=None, super_resolution=None):
        if (type(super_resolution) != type(None)):
            ### used in transferal perceptual loss
            self.learnable_texture_extractor_copy.load_state_dict(
                self.LearnableTextureExtractor.state_dict())
            super_resolution_lv1, super_resolution_lv2, super_resolution_lv3 = self.learnable_texture_extractor_copy(
                (super_resolution + 1.) / 2.)
            return super_resolution_lv1, super_resolution_lv2, super_resolution_lv3

        _, _, upsampled_low_resolution_image_lv3 = self.LearnableTextureExtractor(
            (upsampled_low_resolution_image.detach() + 1.) / 2.)
        _, _, down_and_upsampled_Ref_image_lv3 = self.LearnableTextureExtractor(
            (down_and_upsampled_Ref_image.detach() + 1.) / 2.)

        high_resolution_images_as_references_lv1, high_resolution_images_as_references_lv2, high_resolution_images_as_references_lv3 = self.LearnableTextureExtractor(
            (high_resolution_images_as_references.detach() + 1.) / 2.)

        Soft_attention_map, Hard_attention_output_lv3, Hard_attention_output_lv2, Hard_attention_output_lv1 = self.SearchTransfer(
            upsampled_low_resolution_image_lv3,
            down_and_upsampled_Ref_image_lv3,
            high_resolution_images_as_references_lv1,
            high_resolution_images_as_references_lv2,
            high_resolution_images_as_references_lv3)

        super_resolution = self.MainNet(low_resolution_image,
                                        Soft_attention_map,
                                        Hard_attention_output_lv3,
                                        Hard_attention_output_lv2,
                                        Hard_attention_output_lv1)

        return super_resolution, Soft_attention_map, Hard_attention_output_lv3, Hard_attention_output_lv2, Hard_attention_output_lv1


