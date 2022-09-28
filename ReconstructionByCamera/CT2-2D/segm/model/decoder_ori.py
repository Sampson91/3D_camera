import math
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from einops import rearrange
import numpy as np

from timm.models.layers import trunc_normal_

from segm.model.blocks import Block, FeedForward, Decoder_Block, \
    Decoder_Block_Color, Multiscale_Block
from segm.model.utils import init_weights, CIELAB
from segm.engine import functional_conv2d


def bicubic_upsample(bicubic_upsample, height, width, scaler=2):
    batch, area, channal = bicubic_upsample.size()
    assert area == height * width
    bicubic_upsample = bicubic_upsample.permute(0, 2, 1)
    bicubic_upsample = bicubic_upsample.view(-1, channal, height, width)
    bicubic_upsample = nn.functional.interpolate(bicubic_upsample,
                                                 scale_factor=scaler,
                                                 mode='bicubic')
    batch, channal, height, width = bicubic_upsample.size()
    bicubic_upsample = bicubic_upsample.view(-1, channal, height * width)
    bicubic_upsample = bicubic_upsample.permute(0, 2, 1)
    return bicubic_upsample, height, width


def updown(pooling, height, width):
    batch, area, channal = pooling.size()
    assert area == height * width
    pooling = pooling.permute(0, 2, 1)
    pooling = pooling.view(-1, channal, height, width)
    pooling = nn.functional.interpolate(pooling, scale_factor=4, mode='bicubic')
    pooling = nn.AvgPool2d(4)(pooling)
    batch, channal, height, width = pooling.size()
    pooling = pooling.view(-1, channal, height * width)
    pooling = pooling.permute(0, 2, 1)
    return pooling, height, width


def pixel_upsample(pixel_upsample, height, width):
    batch, area, channal = pixel_upsample.size()
    assert area == height * width
    pixel_upsample = pixel_upsample.permute(0, 2, 1)
    pixel_upsample = pixel_upsample.view(-1, channal, height, width)
    pixel_upsample = nn.PixelShuffle(2)(
        pixel_upsample)  # upsample the resolution and downscale the feature dimension.
    batch, channal, height, width = pixel_upsample.size()
    pixel_upsample = pixel_upsample.view(-1, channal, height * width)
    pixel_upsample = pixel_upsample.permute(0, 2, 1)
    return pixel_upsample, height, width


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv3x3_batch_normalization_relu(in_planes, out_planes, stride=1,
                                     bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block


def conv3x3_relu(in_planes, out_planes, stride=1, bias=True):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.ReLU(inplace=True)
    )
    return block


class DecoderLinear(nn.Module):
    def __init__(self, number_of_colors, patch_size, decoder_encoder):
        super().__init__()

        self.decoder_encoder = decoder_encoder
        self.patch_size = patch_size
        self.number_of_colors = number_of_colors

        self.head = nn.Linear(self.decoder_encoder, number_of_colors)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, linear_decoder, image_size):
        height, width = image_size
        number_patches_along_height = height // self.patch_size
        linear_decoder = self.head(linear_decoder)
        linear_decoder = rearrange(linear_decoder, "b (h w) c -> b c h w",
                                   h=number_patches_along_height)
        # rearrange 是库

        return linear_decoder


class PositionCNN(nn.Module):
    def __init__(self, in_channals, embed_dimension, query_to_ab=None, size=1):
        super(PositionCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channals, embed_dimension, 3, size, 1, bias=True,
                      groups=embed_dimension), )
        self.size = size
        self.query_to_ab = query_to_ab  # [313, 2]

    def forward(self, position_CNN, height, width):
        batch, area, channal = position_CNN.shape  # [B, 313, C]
        if self.query_to_ab is not None:  # color pos.
            cnn_feat = torch.zeros(batch, height, width,
                                   channal).cuda()  # [b, 23, 23, c]
            bin = 10
            torch_ab = torch.from_numpy(self.query_to_ab).cuda()
            # new_ab = (torch_ab + 110) // bin        # [313, 2]
            new_ab = torch.div(torch_ab + 110, bin, rounding_mode='floor')
            cnn_feat[:, new_ab[:, 0].long(), new_ab[:, 1].long(),
            :] = position_CNN  # [B, N, C]

            conv_cnn_feat = self.proj(
                cnn_feat.permute(0, 3, 1, 2))  # [B, C, 23, 23]
            conv_cnn_feat = conv_cnn_feat.permute(0, 2, 3, 1)  # [B, 23, 23, C]
            positionCNN_position = torch.zeros_like(position_CNN)
            positionCNN_position[:, :, :] = conv_cnn_feat[:,
                                            new_ab[:, 0].long(),
                                            new_ab[:, 1].long(), :]  # [B, N, C]
            position_CNN = position_CNN + positionCNN_position
        else:  # patch pos.
            feat_token = position_CNN
            cnn_feat = feat_token.transpose(1, 2).view(batch, channal, height,
                                                       width)
            position_CNN = self.proj(cnn_feat) + cnn_feat
            position_CNN = position_CNN.flatten(2).transpose(1, 2)
        return position_CNN

    def no_weight_decay(self):
        return ['project.%d.weight' % i for i in range(4)]


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, upscale=2):
        super(Upsample, self).__init__()
        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=upscale,
                               stride=upscale),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, need_deconvolution):
        batch, luminance, channal = need_deconvolution.shape
        height = int(math.sqrt(luminance))
        width = int(math.sqrt(luminance))
        need_deconvolution = need_deconvolution.transpose(1,
                                                          2).contiguous().view(
            batch, channal, height, width)
        out = self.deconvolution(need_deconvolution).flatten(2).transpose(1,
                                                                          2).contiguous()  # B H*W C
        return out


class MaskTransformer(nn.Module):
    def __init__(
            self,
            number_of_colors,
            patch_size,
            decoder_encoder,
            number_of_layers,
            number_of_heads,
            number_of_features,
            dimension_feedforward,
            drop_path_rate,
            dropout,
            add_l1_loss,
            color_position,
            change_mask,
            color_as_condition,
            multi_scaled,
            crop_size,
            downchannel,
            add_convolution,
            before_classify,
            l1_convolution,
            l1_linear,
            add_edge,
            number_of_blocks,
            without_colorattn,
            without_colorquery,
            without_classification,
    ):
        super().__init__()
        self.decoder_encoder = decoder_encoder
        self.patch_size = patch_size
        self.number_of_layers = number_of_layers
        self.number_of_colors = number_of_colors
        self.number_of_features = number_of_features
        self.dimension_feedforward = dimension_feedforward
        self.scale = number_of_features ** -0.5
        self.add_l1_loss = add_l1_loss
        self.color_position = color_position
        self.change_mask = change_mask
        self.color_as_condition = color_as_condition
        self.multi_scaled = multi_scaled
        self.downchannel = downchannel
        self.add_convolution = add_convolution
        self.before_classify = before_classify
        self.l1_convolution = l1_convolution
        self.l1_linear = l1_linear
        self.add_edge = add_edge
        self.number_of_blocks = number_of_blocks

        # original two transformer layers.
        drop_rate = [droped.item() for droped in
                     torch.linspace(0, drop_path_rate, number_of_layers)]
        self.blocks = nn.ModuleList(
            [Decoder_Block(number_of_features, number_of_heads,
                           dimension_feedforward, dropout, drop_rate[i]) for i
             in
             range(number_of_layers)]
        )
        if self.color_position:
            self.cielab = CIELAB()
            self.query_to_ab = self.cielab.q_to_ab
            self.cls_embed = nn.Parameter(
                torch.randn(1, number_of_colors, number_of_features))
            self.position_color = nn.ModuleList(
                [PositionCNN(number_of_features, number_of_features,
                             self.query_to_ab)
                 for position in range(2)])
            self.position_patch = nn.ModuleList(
                [PositionCNN(number_of_features, number_of_features) for
                 position in
                 range(2)])
        else:
            self.cls_embed = nn.Parameter(torch.randn(1, number_of_colors,
                                                      number_of_features))  # all learnable
        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(number_of_features, number_of_features))
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(number_of_features, number_of_features))
        self.decoder_norm = nn.LayerNorm(number_of_features)
        if self.add_convolution:
            self.convolutional_layers = nn.ModuleList(
                [nn.Conv2d(number_of_features, number_of_features,
                           kernel_size=3, stride=1, padding=1) for i in
                 range(number_of_layers)]
            )
            self.color_linear = nn.ModuleList(
                [nn.Linear(number_of_features, number_of_features) for i in
                 range(number_of_layers)]
            )
            # use conv to upsample. 16->32->64->128->256, patch: [B, N, 192]
            self.upsampler = nn.Sequential(
                conv3x3_batch_normalization_relu(number_of_features,
                                                 number_of_features),
                nn.ConvTranspose2d(number_of_features, number_of_features,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=True),
                nn.ReLU(True),
                conv3x3_batch_normalization_relu(number_of_features,
                                                 number_of_features),
                nn.ConvTranspose2d(number_of_features, number_of_features,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=True),
                nn.ReLU(True),
                conv3x3_batch_normalization_relu(number_of_features,
                                                 number_of_features),
                nn.ConvTranspose2d(number_of_features, number_of_features,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=True),
                nn.ReLU(True),
                conv3x3_batch_normalization_relu(number_of_features,
                                                 number_of_features),
                nn.ConvTranspose2d(number_of_features, number_of_features,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=True),
            )
        if self.add_edge:
            self.convolution_edge = nn.Conv2d(number_of_features * 2,
                                              number_of_features, kernel_size=3,
                                              stride=1, padding=1, bias=True)

        self.proj_dec = nn.Linear(decoder_encoder, number_of_features)

        if self.add_l1_loss:
            if self.l1_convolution:
                if number_of_features == 1024:
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_batch_normalization_relu(number_of_features,
                                                         number_of_features),
                        nn.ConvTranspose2d(number_of_features,
                                           number_of_features // 4,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 4,
                            number_of_features // 4),
                        nn.ConvTranspose2d(number_of_features // 4,
                                           number_of_features // 16,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 16,
                            number_of_features // 16),
                        nn.ConvTranspose2d(number_of_features // 16,
                                           number_of_features // 64,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 64,
                            number_of_features // 64),
                        nn.ConvTranspose2d(number_of_features // 64,
                                           number_of_features // 256,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        conv3x3(number_of_features // 256, 2)
                    )
                elif number_of_features == 192:
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_batch_normalization_relu(number_of_features,
                                                         number_of_features),
                        nn.ConvTranspose2d(number_of_features,
                                           number_of_features // 2,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 2,
                            number_of_features // 2),
                        nn.ConvTranspose2d(number_of_features // 2,
                                           number_of_features // 4,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 4,
                            number_of_features // 4),
                        nn.ConvTranspose2d(number_of_features // 4,
                                           number_of_features // 8,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 8,
                            number_of_features // 8),
                        nn.ConvTranspose2d(number_of_features // 8,
                                           number_of_features // 8,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        conv3x3(number_of_features // 8, 2)
                    )
                self.tanh = nn.Tanh()
            elif self.l1_linear:
                # self.out_cls = 2 * patch_size * patch_size      # 2 * p * p only for ab channels.
                # self.head = nn.Linear(number_of_features, self.out_cls)
                self.conv_out = conv3x3(number_of_features, 2)
                self.tanh = nn.Tanh()

        self.mask_norm = nn.LayerNorm(number_of_colors)

        self.apply(init_weights)
        trunc_normal_(self.cls_embed, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def calculate_mask(self, mask):
        # mask: [B, 256x256, 313]-> [B, 16x16+313, 16x16+313]
        batch, area, number_of_colors = mask.size()
        height = width = int(math.sqrt(area))  # H=W=256
        process_mask = mask.view(batch, height // self.patch_size,
                                 self.patch_size,
                                 width // self.patch_size, self.patch_size,
                                 number_of_colors)
        process_mask = process_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch, (height // self.patch_size) * (width // self.patch_size),
                   self.patch_size * self.patch_size, number_of_colors)
        process_mask = torch.sum(process_mask, dim=2)
        mask_transpose = process_mask.transpose(1, 2)  # [B, 313, 16x16]
        mask_position = torch.ones(
            (batch, height // self.patch_size * width // self.patch_size,
             height // self.patch_size * width // self.patch_size)).to(
            process_mask.device)
        mask_color = torch.ones(batch, number_of_colors, number_of_colors).to(
            process_mask.device)
        mask_position = torch.cat((mask_position, process_mask),
                               dim=-1)  # [B, 16x16, 16x16+313]
        mask_color = torch.cat((mask_transpose, mask_color),
                               dim=-1)  # [B, 313, 16x16+313]
        process_mask = torch.cat((mask_position, mask_color),
                                 dim=1)  # [B, 16x16+313, 16x16+313]
        return process_mask

    def forward(self, mask_transformer_neural, image_size, input_mask=None,
                patch_pos=None,
                image_luminance=None):
        height, width = image_size
        number_of_patches_along_height = height // self.patch_size
        batch = mask_transformer_neural.size(0)
        mask_transformer_neural = self.proj_dec(mask_transformer_neural)

        # original two transformer layters.
        if self.color_position:
            position_height, position_width = 23, 23
            colors_embed = self.cls_embed.expand(
                mask_transformer_neural.size(0), -1, -1)
            colors_embed = self.position_color[0](colors_embed, position_height,
                                                  position_width)  # cpvt for color tokens.
            mask_transformer_neural = self.position_patch[1](
                mask_transformer_neural, number_of_patches_along_height,
                number_of_patches_along_height)  # cpvt for patch tokens.
        else:
            colors_embed = self.cls_embed.expand(
                mask_transformer_neural.size(0), -1, -1)

        mask_transformer_neural = torch.cat(
            (mask_transformer_neural, colors_embed), 1)
        # process_mask: [B, 256x256, 313]-> [B, 16x16+313, 16x16+313]
        process_mask = self.calculate_mask(input_mask)
        # mask here.
        for block_index in range(self.number_of_layers):
            # input_mask: [B, 256x256, 313]
            mask_transformer_neural = self.blocks[block_index](
                mask_transformer_neural, mask=process_mask)
            if self.add_convolution or self.before_classify:  # 16x16-cls && 256x256-regression, conv after trans.
                patch, color = mask_transformer_neural[:,
                               :-self.number_of_colors], mask_transformer_neural[
                                                         :,
                                                         -self.number_of_colors:]
                patch_height = patch_width = int(math.sqrt(patch.size(1)))
                patch = patch.view(batch, patch_height, patch_width,
                                   self.number_of_features).permute(0, 3, 1,
                                                                    2)  # [B, 192, h, w]

                patch = self.convolutional_layers[block_index](
                    patch)  # conv after per transformer block for patch.
                color = self.color_linear[block_index](
                    color)  # linear after per transformer blocks for color.
                patch = patch.view(batch, self.number_of_features,
                                   patch_height * patch_width).transpose(1, 2)
                if self.color_position and block_index == 0:
                    patch = self.position_patch[1](patch,
                                                   number_of_patches_along_height,
                                                   number_of_patches_along_height)
                    color = self.position_color[1](color, position_height,
                                                   position_width)
                mask_transformer_neural = torch.cat((patch, color), dim=1)

        mask_transformer_neural = self.decoder_norm(
            mask_transformer_neural)

        down_patches, colors_segmentation_feat = mask_transformer_neural[:,
                                                 : -self.number_of_colors], mask_transformer_neural[
                                                                            :,
                                                                            -self.number_of_colors:]
        if self.add_convolution:
            patches = down_patches.view(batch, number_of_patches_along_height,
                                        number_of_patches_along_height,
                                        self.number_of_features).permute(0, 3,
                                                                         1, 2)
            patches = self.upsampler(patches)  # [B, 192, 256, 256]
            patches = patches.view(batch, self.number_of_features,
                                   height * width).transpose(
                1, 2)  # [B, 256x256, 192]

        if self.add_l1_loss:
            if self.before_classify:
                reshape_patch = patches.view(batch,
                                             number_of_patches_along_height,
                                             number_of_patches_along_height,
                                             self.number_of_features).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
                out_feature = self.upsampler(reshape_patch)  # [B, 2, 256, 256]
                out_feature = nn.Tanh()(out_feature)  # normalized to [-1, 1]
            elif self.l1_convolution:
                down_patches = down_patches.view(batch,
                                                 number_of_patches_along_height,
                                                 number_of_patches_along_height,
                                                 self.number_of_features).permute(
                    0, 3, 1, 2)
                out_feature = self.upsampler_l1(
                    down_patches)  # [B, 192, 16, 16]-> [B, 2, 256, 256]
                # reshape_patches = patches.view(B, H, W, self.number_of_features).permute(0, 3, 1, 2)
                # out_feature = self.l1_conv_layer(reshape_patches)  # [B, 2, 256, 256]
                out_feature = self.tanh(out_feature)  # normalized to [-1, 1]
            elif self.l1_linear:
                # out_feature = self.head(down_patches)        # [B, N, 2 * 16 * 16]
                # out_feature = out_feature.contiguous().view(B, GS, GS, 2, self.patch_size, self.patch_size)
                # out_feature = out_feature.permute(0, 3, 1, 4, 2, 5)         # [b, 2, GS, p, GS, p]
                # out_feature = out_feature.contiguous().view(B, 2, GS*self.patch_size, GS*self.patch_size)     # [b, 2, GS*p, GS*P]
                reshape_patch = patches.transpose(1, 2).contiguous().view(batch,
                                                                          self.number_of_features,
                                                                          height,
                                                                          width)
                out_feature = self.conv_out(
                    reshape_patch)  # [B, 2, H, W]
                out_feature = self.tanh(out_feature)  # normalized to [-1, 1]
        else:
            out_feature = None

        patches = patches @ self.proj_patch
        colors_segmentation_feat = colors_segmentation_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        colors_segmentation_feat = colors_segmentation_feat / colors_segmentation_feat.norm(
            dim=-1, keepdim=True)

        masks = patches @ colors_segmentation_feat.transpose(1,
                                                             2)  # [B, 16x16, 313]

        masks = self.mask_norm(masks)  # [B, N, 313]

        # 在LayerNorm之后，softmax之前做mask
        if input_mask is not None:  # add_mask == True
            if self.multi_scaled or self.add_convolution:
                new_mask = input_mask  # [B, 256x256, 313]
            else:
                # new_mask = input_mask[:, :-self.number_of_colors, -self.number_of_colors:]  # [B, N, 313]
                new_mask = process_mask[:, :-self.number_of_colors,
                           -self.number_of_colors:]  # [B, N, 313]
            masks = masks.masked_fill(new_mask == 0,
                                      -float('inf'))  # is it right???

        if self.multi_scaled or self.add_convolution:
            masks = rearrange(masks, "b (h w) n -> b n h w",
                              h=height)  # [B, 313, 256, 256]
        else:
            masks = rearrange(masks, "b (h w) n -> b n h w",
                              h=int(
                                  number_of_patches_along_height))  # [BS,  313, 16, 16]

        return masks, out_feature

    def get_attention_map(self, attention_map, layer_id):
        if layer_id >= self.number_of_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.number_of_layers}."
            )
        attention_map = self.proj_dec(attention_map)
        colors_embed = self.cls_embed.expand(attention_map.size(0), -1, -1)
        attention_map = torch.cat((attention_map, colors_embed), 1)
        for i, block_ in enumerate(self.blocks):
            if i < layer_id:
                attention_map = block_(attention_map)
            else:
                return block_(attention_map, return_attention=True)
