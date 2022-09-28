"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
import time

from segm.model.utils import init_weights, resize_position_embed
from segm.model.blocks import Block

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.vision_transformer import _load_weights


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.number_of_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, image):
        Batch, Channal, Height, Width = image.shape
        embedding = self.proj(image).flatten(2).transpose(1, 2)
        # proj super to nn.Module for using pretrain
        return embedding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        number_of_layers,
        number_of_features,
        dimension_feedforward,
        number_of_heads,
        number_of_colors,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
        partial_finetune=False,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            number_of_features,
            channels,
        )
        self.patch_size = patch_size
        self.number_of_layers = number_of_layers
        self.number_of_features = number_of_features
        self.dimension_feedforward = dimension_feedforward
        self.numer_of_heads = number_of_heads
        self.dropout = nn.Dropout(dropout)
        self.number_of_colors = number_of_colors
        self.partial_finetune = partial_finetune

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, number_of_features))
        # cls_token super to nn.Module for using pretrain
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, number_of_features))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.number_of_patches + 2, number_of_features)
            )
            # pos_embed super to nn.Module for using pretrain
            self.head_dist = nn.Linear(number_of_features, number_of_colors)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.number_of_patches + 1, number_of_features)
            )

        # transformer blocks
        drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, number_of_layers)]
        self.blocks = nn.ModuleList(
            [Block(number_of_features, number_of_heads, dimension_feedforward, dropout, drop_rate[i]) for i in range(number_of_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(number_of_features)
        self.head = nn.Linear(number_of_features, number_of_colors)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)
        print('Successfully load pretrained vit.')
        if self.partial_finetune:
            print('freeze params in blocks.')
            for i in range(6):
                for name, parameters_ in self.blocks[i].named_parameters():
                    parameters_.requires_grad = False


    def forward(self, im, return_features=False, return_position_embed=False):
        batch, _, height, width = im.shape
        patch_size = self.patch_size

        visual_transformer_neural_network = self.patch_embed(im)
        class_tokens = self.cls_token.expand(batch, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(batch, -1, -1)
            visual_transformer_neural_network = torch.cat((class_tokens, dist_tokens, visual_transformer_neural_network), dim=1)
        else:
            visual_transformer_neural_network = torch.cat((class_tokens, visual_transformer_neural_network), dim=1)

        position_embed = self.pos_embed
        number_extra_tokens = 1 + self.distilled
        if visual_transformer_neural_network.shape[1] != position_embed.shape[1]:
            position_embed = resize_position_embed(
                position_embed,
                self.patch_embed.grid_size,
                (height // patch_size, width // patch_size),
                number_extra_tokens,
            )
        visual_transformer_neural_network = visual_transformer_neural_network + position_embed
        visual_transformer_neural_network = self.dropout(visual_transformer_neural_network)

        if self.partial_finetune:
            with torch.no_grad():
                for block_number in range(6):
                    visual_transformer_neural_network = self.blocks[block_number](visual_transformer_neural_network)
            for block_number in range(6, len(self.blocks)):
                visual_transformer_neural_network = self.blocks[block_number](visual_transformer_neural_network)
        else:
            for block in self.blocks:
                visual_transformer_neural_network = block(visual_transformer_neural_network)

        visual_transformer_neural_network = self.norm(visual_transformer_neural_network)

        if return_features:
            if return_position_embed:
                return visual_transformer_neural_network, position_embed
            else:
                return visual_transformer_neural_network, None

        if self.distilled:
            visual_transformer_neural_network, visual_transformer_neural_network_dist = visual_transformer_neural_network[:, 0], visual_transformer_neural_network[:, 1]
            visual_transformer_neural_network = self.head(visual_transformer_neural_network)
            visual_transformer_neural_network_dist = self.head_dist(visual_transformer_neural_network_dist)
            visual_transformer_neural_network = (visual_transformer_neural_network + visual_transformer_neural_network_dist) / 2
        else:
            visual_transformer_neural_network = visual_transformer_neural_network[:, 0]
            visual_transformer_neural_network = self.head(visual_transformer_neural_network)
        return visual_transformer_neural_network

    def get_attention_map(self, image, layer_id):
        if layer_id >= self.number_of_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.number_of_layers}."
            )
        batch, _, height, width = image.shape
        patch_size = self.patch_size

        embed_image = self.patch_embed(image)
        class_token = self.cls_token.expand(batch, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(batch, -1, -1)
            embed_image = torch.cat((class_token, dist_tokens, embed_image), dim=1)
        else:
            embed_image = torch.cat((class_token, embed_image), dim=1)

        position_embed = self.pos_embed
        number_of_extra_tokens = 1 + self.distilled
        if embed_image.shape[1] != position_embed.shape[1]:
            position_embed = resize_position_embed(
                position_embed,
                self.patch_embed.grid_size,
                (height // patch_size, width // patch_size),
                number_of_extra_tokens,
            )
        embed_image = embed_image + position_embed

        for i, block_ in enumerate(self.blocks):
            if i < layer_id:
                embed_image = block_(embed_image)
            else:
                return block_(embed_image, return_attention=True)


