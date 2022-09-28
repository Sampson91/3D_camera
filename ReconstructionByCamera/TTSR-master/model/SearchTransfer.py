import math
import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional

from option import args
class SearchTransfer(torch_neural_network.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i != dim else -1 for i in
                                   range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, upsampled_low_resolution_image_lv3,
                down_and_upsampled_Ref_image_lv3,
                high_resolution_images_as_references_lv1,
                high_resolution_images_as_references_lv2,
                high_resolution_images_as_references_lv3):
        ### search
        # nn.unfold(kernel_size=(k,k)): input [b c h w] -> [b c*k*k h*w]
        upsampled_low_resolution_image_lv3_unfold = torch_neural_network_functional.unfold(
            upsampled_low_resolution_image_lv3, kernel_size=(3, 3),
            padding=1)  # Q -> [N, C*k*k, H*W] 
        down_and_upsampled_Ref_image_lv3_unfold = torch_neural_network_functional.unfold(
            down_and_upsampled_Ref_image_lv3, kernel_size=(3, 3),
            padding=1)  # K -> [N, C*k*k, Hr*Wr]

        # permute(0,2,1) : [N, C*k*k, Hr*Wr] -> [N, Hr*Wr, C*k*k]
        down_and_upsampled_Ref_image_lv3_unfold = down_and_upsampled_Ref_image_lv3_unfold.permute(
            0, 2, 1)

        down_and_upsampled_Ref_image_lv3_unfold = torch_neural_network_functional.normalize(
            down_and_upsampled_Ref_image_lv3_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        upsampled_low_resolution_image_lv3_unfold = torch_neural_network_functional.normalize(
            upsampled_low_resolution_image_lv3_unfold, dim=1)  # [N, C*k*k, H*W]

        # relevance embedding
        # a size [N ha wa] ; b size [N hb wb]
        # e = torch.bmm(a,b) 一维相同的矩阵乘法 --> e size [N ha wb]
        relevance_embedding_lv3 = torch.bmm(
            down_and_upsampled_Ref_image_lv3_unfold,
            upsampled_low_resolution_image_lv3_unfold)  # [N, Hr*Wr, H*W]
        # relevance_embedding_lv3_star: for soft-attention
        # relevance_embedding_lv3_star_arg: for hard-attention
        relevance_embedding_lv3_star, relevance_embedding_lv3_star_arg = torch.max(
            relevance_embedding_lv3, dim=1)  # [N, H*W]

        ### transfer -V
        high_resolution_images_as_references_lv3_unfold = torch_neural_network_functional.unfold(
            high_resolution_images_as_references_lv3, kernel_size=(3, 3),
            padding=1)
        high_resolution_images_as_references_lv2_unfold = torch_neural_network_functional.unfold(
            high_resolution_images_as_references_lv2, kernel_size=(6, 6),
            padding=2, stride=2)
        high_resolution_images_as_references_lv1_unfold = torch_neural_network_functional.unfold(
            high_resolution_images_as_references_lv1, kernel_size=(12, 12),
            padding=4, stride=4)

        Hard_attention_output_lv3_unfold = self.bis(
            high_resolution_images_as_references_lv3_unfold, 2,
            relevance_embedding_lv3_star_arg)
        Hard_attention_output_lv2_unfold = self.bis(
            high_resolution_images_as_references_lv2_unfold, 2,
            relevance_embedding_lv3_star_arg)
        Hard_attention_output_lv1_unfold = self.bis(
            high_resolution_images_as_references_lv1_unfold, 2,
            relevance_embedding_lv3_star_arg)

        Hard_attention_output_lv3 = torch_neural_network_functional.fold(
            Hard_attention_output_lv3_unfold,
            output_size=upsampled_low_resolution_image_lv3.size()[-2:],
            kernel_size=(3, 3), padding=1) / (3. * 3.)
        Hard_attention_output_lv2 = torch_neural_network_functional.fold(
            Hard_attention_output_lv2_unfold, output_size=(
                upsampled_low_resolution_image_lv3.size(2) * 2,
                upsampled_low_resolution_image_lv3.size(3) * 2),
            kernel_size=(6, 6),
            padding=2, stride=2) / (3. * 3.)
        Hard_attention_output_lv1 = torch_neural_network_functional.fold(
            Hard_attention_output_lv1_unfold, output_size=(
                upsampled_low_resolution_image_lv3.size(2) * 4,
                upsampled_low_resolution_image_lv3.size(3) * 4),
            kernel_size=(12, 12), padding=4, stride=4) / (3. * 3.)

        Soft_attention_map = relevance_embedding_lv3_star.view(
            relevance_embedding_lv3_star.size(0), 1,
            upsampled_low_resolution_image_lv3.size(2),
            upsampled_low_resolution_image_lv3.size(3))

        return Soft_attention_map, Hard_attention_output_lv3, Hard_attention_output_lv2, Hard_attention_output_lv1
