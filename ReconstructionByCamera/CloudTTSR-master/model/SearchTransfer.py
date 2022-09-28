import math
import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional


class SearchTransfer(torch_neural_network.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        self.lv3 = torch_neural_network.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lv2 = torch_neural_network.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.lv1 = torch_neural_network.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

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
        
        # transfer the dimension for cloud form 3D to 4D
        upsampled_low_resolution_image_lv3 = torch.unsqueeze(upsampled_low_resolution_image_lv3, 2)
        down_and_upsampled_Ref_image_lv3 = torch.unsqueeze(down_and_upsampled_Ref_image_lv3, 2)
        high_resolution_images_as_references_lv1 = torch.unsqueeze(high_resolution_images_as_references_lv1, 2)
        high_resolution_images_as_references_lv2 = torch.unsqueeze(high_resolution_images_as_references_lv2, 2)
        high_resolution_images_as_references_lv3 = torch.unsqueeze(high_resolution_images_as_references_lv3, 2)
        # search
        upsampled_low_resolution_image_lv3_unfold = torch_neural_network_functional.unfold(
            upsampled_low_resolution_image_lv3, kernel_size=(3, 3),
            padding=1)  # Q
        down_and_upsampled_Ref_image_lv3_unfold = torch_neural_network_functional.unfold(
            down_and_upsampled_Ref_image_lv3, kernel_size=(3, 3),
            padding=1)  # K
        down_and_upsampled_Ref_image_lv3_unfold = down_and_upsampled_Ref_image_lv3_unfold.permute(
            0, 2, 1)

        down_and_upsampled_Ref_image_lv3_unfold = torch_neural_network_functional.normalize(
            down_and_upsampled_Ref_image_lv3_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        upsampled_low_resolution_image_lv3_unfold = torch_neural_network_functional.normalize(
            upsampled_low_resolution_image_lv3_unfold, dim=1)  # [N, C*k*k, H*W]

        # relevance embedding
        relevance_embedding_lv3 = torch.bmm(
            down_and_upsampled_Ref_image_lv3_unfold,
            upsampled_low_resolution_image_lv3_unfold)  # [N, Hr*Wr, H*W]
        # relevance_embedding_lv3_star: for soft-attention
        # relevance_embedding_lv3_star_arg: for hard-attention
        relevance_embedding_lv3_star, relevance_embedding_lv3_star_arg = torch.max(
            relevance_embedding_lv3, dim=1)  # [N, H*W]

        # Chnage unfold and fold to Conv1D
        Hard_attention_output_lv3 = self.lv3(high_resolution_images_as_references_lv3)
        Hard_attention_output_lv2 = self.lv2(high_resolution_images_as_references_lv2)
        Hard_attention_output_lv1 = self.lv1(high_resolution_images_as_references_lv1)

        Soft_attention_map = relevance_embedding_lv3_star.view(
            relevance_embedding_lv3_star.size(0), 1,
            upsampled_low_resolution_image_lv3.size(2),
            upsampled_low_resolution_image_lv3.size(3))

        # transfer the dimension for cloud form 4D to 3D, need to ensure the dimesion of 1 is 1
        Soft_attention_map = torch.squeeze(Soft_attention_map, 2)
        Hard_attention_output_lv3 = torch.squeeze(Hard_attention_output_lv3, 2)
        Hard_attention_output_lv2 = torch.squeeze(Hard_attention_output_lv2, 2)
        Hard_attention_output_lv1 = torch.squeeze(Hard_attention_output_lv1, 2)

        return Soft_attention_map, Hard_attention_output_lv3, Hard_attention_output_lv2, Hard_attention_output_lv1
