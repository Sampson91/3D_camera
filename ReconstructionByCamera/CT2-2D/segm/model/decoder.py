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


def bicubic_upsample(bicubic_upsample, Height, Width, scaler=2):
    batch, area, channal = bicubic_upsample.size()
    assert area == Height * Width
    bicubic_upsample = bicubic_upsample.permute(0, 2, 1)
    bicubic_upsample = bicubic_upsample.view(-1, channal, Height, Width)
    bicubic_upsample = nn.functional.interpolate(bicubic_upsample,
                                                 scale_factor=scaler,
                                                 mode='bicubic')
    batch, channal, Height, Width = bicubic_upsample.size()
    bicubic_upsample = bicubic_upsample.view(-1, channal, Height * Width)
    bicubic_upsample = bicubic_upsample.permute(0, 2, 1)
    return bicubic_upsample, Height, Width


def updown(pooling, Height, Width):
    batch, area, channal = pooling.size()
    assert area == Height * Width
    pooling = pooling.permute(0, 2, 1)
    pooling = pooling.view(-1, channal, Height, Width)
    pooling = nn.functional.interpolate(pooling, scale_factor=4, mode='bicubic')
    pooling = nn.AvgPool2d(4)(pooling)
    batch, channal, Height, Width = pooling.size()
    pooling = pooling.view(-1, channal, Height * Width)
    pooling = pooling.permute(0, 2, 1)
    return pooling, Height, Width


def pixel_upsample(pixel_upsample, Height, Width):
    batch, area, channal = pixel_upsample.size()
    assert area == Height * Width
    pixel_upsample = pixel_upsample.permute(0, 2, 1)
    pixel_upsample = pixel_upsample.view(-1, channal, Height, Width)
    pixel_upsample = nn.PixelShuffle(2)(
        pixel_upsample)  # upsample the resolution and downscale the feature dimension.
    batch, channal, Height, Width = pixel_upsample.size()
    pixel_upsample = pixel_upsample.view(-1, channal, Height * Width)
    pixel_upsample = pixel_upsample.permute(0, 2, 1)
    return pixel_upsample, Height, Width


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

    def forward(self, linear_decode, image_size):
        Height, Width = image_size
        height_by_patch_size = Height // self.patch_size
        linear_decode = self.head(linear_decode)
        linear_decode = rearrange(linear_decode, "b (h w) c -> b c h w",
                                  h=height_by_patch_size)
        # "b (h w) c -> b c h w" 和 h rearrange 库里的

        return linear_decode


class PositionCNN(nn.Module):
    def __init__(self, in_channals, embed_dimension, query_to_ab=None, size=1):
        super(PositionCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channals, embed_dimension, 3, size, 1, bias=True,
                      groups=embed_dimension), )
        self.size = size
        self.query_to_ab = query_to_ab  # [313, 2]

    def forward(self, position_CNN, Height, Width):
        Batch, Number_of_dimentions, Channal = position_CNN.shape  # [B, 313, C]
        if self.query_to_ab is not None:  # color pos.
            cnn_feat = torch.zeros(Batch, Height, Width, Channal).to(
                position_CNN.device)  # [b, 23, 23, c]
            bin = 10
            torch_ab = torch.from_numpy(self.query_to_ab).to(
                position_CNN.device)
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
            cnn_feat = feat_token.transpose(1, 2).contiguous().view(Batch,
                                                                    Channal,
                                                                    Height,
                                                                    Width)
            position_CNN = self.proj(cnn_feat) + cnn_feat
            position_CNN = position_CNN.flatten(2).transpose(1, 2)
        return position_CNN

    def no_weight_decay(self):
        return ['project.%d.weight' % i for i in range(4)]


class Sin_Position(nn.Module):
    def __init__(self, number_of_features, dimension):
        super(Sin_Position, self).__init__()
        self.dimension = dimension
        self.number_of_features = number_of_features

    def forward(self, sin_position_neural):
        if self.dimension == 1:
            # for color tokens.
            batch, area, channal = sin_position_neural.shape
            position_embedding = torch.zeros(area, channal).to(
                sin_position_neural.device)  # [N, C]
            position = torch.arange(0, area).unsqueeze(1)  # [N, 1]
            division_term = torch.exp(
                (torch.arange(0, channal, dtype=torch.float) * -(
                        math.log(10000.0) / channal)))
            position_embedding[:, 0::2] = torch.sin(
                position.float() * division_term)
            position_embedding[:, 1::2] = torch.cos(
                position.float() * division_term)
            position_embedding = position_embedding.unsqueeze(0).repeat(
                batch)  # [B, N, C]
            sin_position_neural += position_embedding
        elif self.dimension == 2:
            batch, area, channal = sin_position_neural.shape
            height = width = math.sqrt(area)  # 16
            patch_token = sin_position_neural.transpose(1, 2).contiguous().view(
                batch, channal, height, width)
            position_embedding = torch.zeros(channal, height, width).to(
                sin_position_neural.device)
            number_of_features = channal // 2
            division_term = torch.exp(
                torch.arange(0, number_of_features, 2) * -math.log(
                    10000.0) / number_of_features)
            pos_w = torch.arange(0., width).unsqueeze(1)
            pos_h = torch.arange(0., height).unsqueeze(1)
            position_embedding[0:number_of_features:2, :, :] = torch.sin(
                pos_w * division_term).transpose(0, 1).unsqueeze(1).repeat(1,
                                                                           height,
                                                                           1)
            position_embedding[1:number_of_features:2, :, :] = torch.cos(
                pos_w * division_term).transpose(0, 1).unsqueeze(1).repeat(1,
                                                                           height,
                                                                           1)
            position_embedding[number_of_features::2, :, :] = torch.sin(
                pos_h * division_term).transpose(0, 1).unsqueeze(2).repeat(1, 1,
                                                                           width)
            position_embedding[number_of_features + 1::2, :, :] = torch.cos(
                pos_h * division_term).transpose(0, 1).unsqueeze(2).repeat(1, 1,
                                                                           width)
            position_embedding = position_embedding.unsqueeze(0).repeat(batch)
            patch_token += position_embedding
            sin_position_neural = patch_token.flatten(2).transpose(1,
                                                                   2)  # [B, N, C]
        return sin_position_neural


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
        need_deconvolution = need_deconvolution.transpose(1, 2).contiguous().view(batch, channal, height, width)
        out = self.deconvolution(need_deconvolution).flatten(2).transpose(1, 2).contiguous()  # B H*W C
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
            sin_color_position,
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
        self.without_colorattn = without_colorattn
        self.without_colorquery = without_colorquery
        self.without_classification = without_classification
        self.sin_color_position = sin_color_position
        drop_rate = [droped.item() for droped in
                     torch.linspace(0, drop_path_rate, number_of_layers)]
        self.per_block = nn.ModuleList(
            [Decoder_Block(number_of_features, number_of_heads,
                           dimension_feedforward, dropout, drop_rate[i]) for i
             in range(number_of_layers)]
        )

        self.blocks = nn.ModuleList(
            [self.per_block for i in range(number_of_blocks)])

        if self.color_position:
            self.cielab = CIELAB()
            self.query_to_ab = self.cielab.q_to_ab
            self.cls_emb = nn.Parameter(
                torch.randn(1, number_of_colors, number_of_features))
            if self.sin_color_position:
                self.pos_color = Sin_Position(number_of_features, dimension=1)
                self.pos_patch = Sin_Position(number_of_features, dimension=2)
            else:
                self.pos_color = PositionCNN(number_of_features,
                                             number_of_features,
                                             self.query_to_ab)
                self.pos_patch = PositionCNN(number_of_features,
                                             number_of_features)
        else:
            self.cls_emb = nn.Parameter(torch.randn(1, number_of_colors,
                                                    number_of_features))  # all learnable
        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(number_of_features, number_of_features))
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(number_of_features, number_of_features))
        self.decoder_norm = nn.LayerNorm(number_of_features)
        if self.add_convolution:
            self.conv_layers = nn.ModuleList(
                [nn.Conv2d(number_of_features, number_of_features,
                           kernel_size=3, stride=1, padding=1) for i in
                 range(number_of_blocks)]
            )
            self.color_linear = nn.ModuleList(
                [nn.Linear(number_of_features, number_of_features) for i in
                 range(number_of_blocks)]
            )
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
                if number_of_features == 192:  # tiny
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_batch_normalization_relu(number_of_features,
                                                         number_of_features),
                        nn.ConvTranspose2d(number_of_features,
                                           number_of_features // 2,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 2, number_of_features // 2),
                        nn.ConvTranspose2d(number_of_features // 2,
                                           number_of_features // 4,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 4, number_of_features // 4),
                        nn.ConvTranspose2d(number_of_features // 4,
                                           number_of_features // 8,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 8, number_of_features // 8),
                        nn.ConvTranspose2d(number_of_features // 8,
                                           number_of_features // 8,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        conv3x3(number_of_features // 8, 2)
                    )
                elif number_of_features == 384 or number_of_features == 768:  # small
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_batch_normalization_relu(number_of_features,
                                                         number_of_features),
                        nn.ConvTranspose2d(number_of_features,
                                           number_of_features // 2,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 2, number_of_features // 2),
                        nn.ConvTranspose2d(number_of_features // 2,
                                           number_of_features // 4,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 4, number_of_features // 4),
                        nn.ConvTranspose2d(number_of_features // 4,
                                           number_of_features // 8,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 8, number_of_features // 8),
                        nn.ConvTranspose2d(number_of_features // 8,
                                           number_of_features // 16,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        conv3x3(number_of_features // 16, 2))
                elif number_of_features == 1024:  # Large or base
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_batch_normalization_relu(number_of_features,
                                                         number_of_features),
                        nn.ConvTranspose2d(number_of_features,
                                           number_of_features // 4,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 4, number_of_features // 4),
                        nn.ConvTranspose2d(number_of_features // 4,
                                           number_of_features // 16,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 16, number_of_features // 16),
                        nn.ConvTranspose2d(number_of_features // 16,
                                           number_of_features // 64,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        nn.ReLU(True),
                        conv3x3_batch_normalization_relu(
                            number_of_features // 64, number_of_features // 64),
                        nn.ConvTranspose2d(number_of_features // 64,
                                           number_of_features // 256,
                                           kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        conv3x3(number_of_features // 256, 2)
                    )

                self.tanh = nn.Tanh()
            elif self.l1_linear:
                self.conv_out = conv3x3(number_of_features, 2)
                self.tanh = nn.Tanh()

        self.mask_norm = nn.LayerNorm(number_of_colors)
        if self.without_colorquery:
            self.classifier = nn.Linear(number_of_features,
                                        self.number_of_colors)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def calculate_mask(self, mask):
        # mask: [B, 256x256, 313]-> [B, 16x16+313, 16x16+313]
        batch, area, number_of_colors = mask.size()
        height = width = int(math.sqrt(area))  # H=W=256
        process_mask = mask.view(batch, height // self.patch_size, self.patch_size,
                                 width // self.patch_size, self.patch_size,
                                 number_of_colors)
        process_mask = process_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch, (height // self.patch_size) * (width // self.patch_size),
               self.patch_size * self.patch_size, number_of_colors)
        process_mask = torch.sum(process_mask, dim=2)
        mask_transpose = process_mask.transpose(1, 2)  # [B, 313, 16x16]
        mask_position = torch.ones((batch, height // self.patch_size * width // self.patch_size,
                             height // self.patch_size * width // self.patch_size)).to(
            process_mask.device)
        mask_color = torch.ones(batch, number_of_colors, number_of_colors).to(
            process_mask.device)
        mask_position = torch.cat((mask_position, process_mask),
                           dim=-1)  # [B, 16x16, 16x16+313]
        mask_color = torch.cat((mask_transpose, mask_color), dim=-1)  # [B, 313, 16x16+313]
        process_mask = torch.cat((mask_position, mask_color),
                                 dim=1)  # [B, 16x16+313, 16x16+313]
        return process_mask

    def forward(self, mask_transformer_neural, image_size, input_mask=None, patch_pos=None,
                image_luminance=None):
        height, width = image_size
        number_of_patches_along_height = height // self.patch_size
        batch = mask_transformer_neural.size(0)
        mask_transformer_neural = self.proj_dec(mask_transformer_neural)
        if self.color_position:
            if self.sin_color_position:
                colors_emb = self.cls_emb.expand(mask_transformer_neural.size(0), -1, -1)
                colors_emb = self.pos_color(colors_emb)
                mask_transformer_neural = self.pos_patch(mask_transformer_neural)
            else:
                position_height, position_width = 23, 23
                colors_emb = self.cls_emb.expand(mask_transformer_neural.size(0), -1, -1)
                colors_emb = self.pos_color(colors_emb, position_height,
                                         position_width)  # cpvt for color tokens.
                mask_transformer_neural = self.pos_patch(mask_transformer_neural, number_of_patches_along_height, number_of_patches_along_height)  # cpvt for patch tokens.
        else:
            colors_emb = self.cls_emb.expand(mask_transformer_neural.size(0), -1, -1)

        mask_transformer_neural = torch.cat((mask_transformer_neural, colors_emb), 1)
        if input_mask is not None:
            process_mask = self.calculate_mask(input_mask)
        else:
            process_mask = None
        for block_index in range(self.number_of_blocks):
            for layer_index in range(self.number_of_layers):
                mask_transformer_neural = self.blocks[block_index][layer_index](mask_transformer_neural, mask=process_mask,
                                                                            without_colorattn=self.without_colorattn)

            if self.add_convolution:  # 16x16-cls && 256x256-regression, conv after trans.
                patch, color = mask_transformer_neural[:, :-self.number_of_colors], mask_transformer_neural[:,
                                                              -self.number_of_colors:]
                patch_h = patch_w = int(math.sqrt(patch.size(1)))
                patch = patch.contiguous().view(batch, patch_h, patch_w,
                                                self.number_of_features).permute(
                    0, 3, 1, 2)  # [B, 192, h, w]
                patch = self.conv_layers[block_index](
                    patch).contiguous()  # conv after per transformer block for patch.
                color = self.color_linear[block_index](
                    color)  # linear after per transformer blocks for color.
                patch = patch.view(batch, self.number_of_features,
                                   patch_h * patch_w).transpose(1, 2)
                mask_transformer_neural = torch.cat((patch, color), dim=1)

        mask_transformer_neural = self.decoder_norm(mask_transformer_neural)

        down_patches, colors_segmentation_feat = mask_transformer_neural[:, : -self.number_of_colors], mask_transformer_neural[:,
                                                                     -self.number_of_colors:]
        if self.add_convolution and not self.without_classification:  # default.
            patches = down_patches.contiguous().view(batch, number_of_patches_along_height, number_of_patches_along_height,
                                                     self.number_of_features).permute(
                0, 3, 1, 2)
            patches = self.upsampler(patches).contiguous()  # [B, 192, 256, 256]
            patches = patches.view(batch, self.number_of_features, height * width).transpose(
                1, 2).contiguous()  # [B, 256x256, 192]

        if self.add_l1_loss:
            if self.before_classify:
                reshape_patch = patches.view(batch, number_of_patches_along_height, number_of_patches_along_height,
                                             self.number_of_features).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
                out_feature = self.upsampler(reshape_patch)  # [B, 2, 256, 256]
                out_feature = nn.Tanh()(out_feature)  # normalized to [-1, 1]
            elif self.l1_convolution:
                down_patches = down_patches.contiguous().view(batch, number_of_patches_along_height, number_of_patches_along_height,
                                                              self.number_of_features).permute(
                    0, 3, 1, 2)
                out_feature = self.upsampler_l1(
                    down_patches)  # [B, 192, 16, 16]-> [B, 2, 256, 256]
                out_feature = self.tanh(out_feature)  # normalized to [-1, 1]
            elif self.l1_linear:
                reshape_patch = patches.transpose(1, 2).contiguous().view(batch,
                                                                          self.number_of_features,
                                                                          height, width)
                out_feature = self.conv_out(reshape_patch)  # [B, 2, H, W]
                out_feature = self.tanh(out_feature)  # normalized to [-1, 1]
        else:
            out_feature = None

        if not self.without_colorquery and not self.without_classification:  # default.
            patches = patches @ self.proj_patch
            colors_segmentation_feat = colors_segmentation_feat @ self.proj_classes

            patches = patches / patches.norm(dim=-1, keepdim=True)
            colors_segmentation_feat = colors_segmentation_feat / colors_segmentation_feat.norm(dim=-1,
                                                                                                keepdim=True)

            masks = patches @ colors_segmentation_feat.transpose(1, 2)  # [B, 16x16, 313]
            masks = self.mask_norm(masks)  # [B, N, 313]
            if input_mask is not None:  # add_mask == True
                if self.multi_scaled or self.add_convolution:
                    new_mask = input_mask  # [B, 256x256, 313]
                else:
                    new_mask = process_mask[:, :-self.number_of_colors,
                               -self.number_of_colors:]  # [B, N, 313]
                masks = masks.masked_fill(new_mask == 0,
                                          -float('inf'))  # is it right???
        elif self.without_colorquery:
            assert self.without_colorquery is True
            masks = self.classifier(
                patches)  # [B, 256x256, 192]-> [B, 256x256, 313]
        elif self.without_classification:
            assert self.without_classification is True
            return None, out_feature

        if self.multi_scaled or self.add_convolution:
            masks = rearrange(masks, "b (h w) n -> b n h w",
                              h=height)  # [B, 313, 256, 256]

        return masks, out_feature

    def get_attention_map(self, attention_map, layer_id):
        if layer_id >= self.number_of_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.number_of_layers}."
            )
        attention_map = self.proj_dec(attention_map)
        colors_embed = self.cls_emb.expand(attention_map.size(0), -1, -1)
        attention_map = torch.cat((attention_map, colors_embed), 1)
        for i, block_ in enumerate(self.blocks):
            if i < layer_id:
                attention_map = block_(attention_map)
            else:
                return block_(attention_map, return_attention=True)
