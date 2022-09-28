"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
import time
import torch.nn.functional as torch_functional
import numpy as np
from timm.models.layers import DropPath



class FeedForward(nn.Module):
    def __init__(self, dimension, hidden_dimension, dropout, out_dimension=None):
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden_dimension)
        # fc1 super to nn.Module as key to use vit pretrain
        self.act = nn.GELU()
        if out_dimension is None:
            out_dimension = dimension
        self.fc2 = nn.Linear(hidden_dimension, out_dimension)
        # fc1 super to nn.Module as key to use vit pretrain
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, multilayer_perceptron):
        multilayer_perceptron = self.fc1(multilayer_perceptron)
        multilayer_perceptron = self.act(multilayer_perceptron)
        multilayer_perceptron = self.drop(multilayer_perceptron)
        multilayer_perceptron = self.fc2(multilayer_perceptron)
        multilayer_perceptron = self.drop(multilayer_perceptron)
        return multilayer_perceptron


class Attention(nn.Module):         # for vit attention
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dimension = dimension // heads
        self.scale = head_dimension ** -0.5
        self.attention = None

        self.qkv = nn.Linear(dimension, dimension * 3)
        self.attention_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dimension, dimension)
        # proj super to nn.Module as key to use vit pretrain
        self.proj_drop = nn.Dropout(dropout)
        # proj_drop super to nn.Module as key to use vit pretrain

    @property
    def unwrapped(self):
        return self

    def forward(self, attention_neural, mask=None):
        batch, area_plus_colors, channal = attention_neural.shape       # x: [B, 16*16+313, C]
        qkv = (
            self.qkv(attention_neural)
            .reshape(batch, area_plus_colors, 3, self.heads, channal // self.heads)
            .permute(2, 0, 3, 1, 4)     # [3, B, self.heads, N, C//heads]
        )
        query, key, value = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attention = (query @ key.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]
        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention)

        attention_neural = (attention @ value).transpose(1, 2).reshape(batch, area_plus_colors, channal)
        attention_neural = self.proj(attention_neural)
        attention_neural = self.proj_drop(attention_neural)

        return attention_neural, attention


class Block(nn.Module):
    def __init__(self, dimension, heads, multilayer_perceptron_dimension, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.attn = Attention(dimension, heads, dropout)
        self.mlp = FeedForward(dimension, multilayer_perceptron_dimension, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # all above super to nn.Module as key to use vit pretrain

    def forward(self, need_normalization, mask=None, return_attention=False):
        attention_neural, attention = self.attn(self.norm1(need_normalization), mask)
        if return_attention:
            return attention
        need_normalization = need_normalization + self.drop_path(attention_neural)
        need_normalization = need_normalization + self.drop_path(self.mlp(self.norm2(need_normalization)))
        return need_normalization


class Decoder_Attention(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dimension = dimension // heads
        self.scale = head_dimension ** -0.5
        self.attention = None
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.attention_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dimension, dimension)
        self.project_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, decoder_attention_neural, mask=None, without_colorattn=False):
        if not without_colorattn:
            batch, area_plus_colors, channal = decoder_attention_neural.shape       # x: [B, 16*16+313, C]
            qkv = (
                self.qkv(decoder_attention_neural)
                .reshape(batch, area_plus_colors, 3, self.heads, channal // self.heads)
                .permute(2, 0, 3, 1, 4)     # [3, B, self.heads, N, C//heads]
            )
            query, key, value = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            # q,k,v: [B, heads, 16*16+313, C//heads] , heads = 3

            attention = (query @ key.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]

            if mask is not None:        # add_mask == True
                expand_mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)     # [B, heads, 16*16+313, 16*16+313]
                attention = attention.masked_fill(expand_mask == 0, -float('inf'))

            attention = attention.softmax(dim=-1)
            attention = self.attention_drop(attention)

            decoder_attention_neural = (attention @ value).transpose(1, 2).reshape(batch, area_plus_colors, channal)
            decoder_attention_neural = self.proj(decoder_attention_neural)
            decoder_attention_neural = self.project_drop(decoder_attention_neural)
        else:
            assert without_colorattn is True
            batch, area_plus_colors, channal = decoder_attention_neural.shape
            patch_number, number_of_colors = 16*16, 313
            patch_tokens, color_tokens = decoder_attention_neural[:, :-number_of_colors, :], decoder_attention_neural[:, patch_number:, :]
            qkv = (self.qkv(patch_tokens).reshape(batch, patch_number, 3, self.heads, channal // self.heads).permute(2, 0, 3, 1, 4))
            query, key, value = (qkv[0], qkv[1], qkv[2])
            attention = (query @ key.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim=-1)
            attention = self.attention_drop(attention)
            patches = (attention @ value).transpose(1, 2).reshape(batch, patch_number, channal)
            patches = self.proj(patches)
            patches = self.project_drop(patches)
            decoder_attention_neural = torch.cat((patches, color_tokens), dim=1)       # [B, N, C]

        return decoder_attention_neural, attention


class Decoder_Block(nn.Module):
    def __init__(self, dimension, heads, multilayer_perceptron_dimension, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.attn = Decoder_Attention(dimension, heads, dropout)
        self.mlp = FeedForward(dimension, multilayer_perceptron_dimension, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # all above super to nn.Module as key to use vit pretrain

    def forward(self, decoder_neural, mask=None, return_attention=False, without_colorattn=False):
        need_drop, attention = self.attn(self.norm1(decoder_neural), mask, without_colorattn=without_colorattn)
        if return_attention:
            return attention
        decoder_neural = decoder_neural + self.drop_path(need_drop)
        decoder_neural = decoder_neural + self.drop_path(self.mlp(self.norm2(decoder_neural)))

        return decoder_neural


################################## For multi-scaled Decoder ###############################
class PixelNormalization(nn.Module):
    def __init__(self, dimension):
        super().__init__()
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


class Multidecoder_Attention(nn.Module):
    def __init__(self, dimension, heads, dropout, number_of_windows):
        super().__init__()
        self.heads = heads
        head_dimension = dimension // heads
        self.scale = head_dimension ** -0.5
        self.attention = None
        self.number_of_windows = number_of_windows

        self.qkv = nn.Linear(dimension, dimension * 3)
        self.attention_drop = nn.Dropout(dropout)
        self.project = nn.Linear(dimension, dimension)
        self.project_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, patch_tokens, color_tokens, mask):
        batch, area, channal = patch_tokens.shape
        number_of_colors = color_tokens.size(1)
        batchsize = batch//self.number_of_windows
        patch_tokens = patch_tokens.view(batchsize, self.number_of_windows, area, channal)
        new_patch_tokens = torch.zeros_like(patch_tokens)
        mask = mask.view(batchsize, self.number_of_windows, area + number_of_colors, area + number_of_colors)
        for win in range(self.number_of_windows):
            catted_shape = torch.cat((patch_tokens[:, win], color_tokens), dim=1)      #[batchsize, N+313, C]
            qkv = (self.qkv(catted_shape).reshape(batchsize, area+number_of_colors, 3, self.heads, channal//self.heads).permute(2, 0, 3, 1, 4))
            query, key, value = (qkv[0], qkv[1], qkv[2])      # [batchsize, heads, 16*16+313, C//heads]
            attention = (query @ key.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]
            expand_mask = mask[:, win].unsqueeze(1).repeat(1, self.heads, 1, 1)     # [batchsize, heads, N+313, N+313]
            attention = attention.masked_fill(expand_mask == 0, -float('inf'))
            attention = attention.softmax(dim=-1)
            attention = self.attention_drop(attention)
            catted_shape = (attention @ value).transpose(1, 2).reshape(batchsize, area+number_of_colors, channal)
            catted_shape = self.project(catted_shape)
            catted_shape = self.project_drop(catted_shape)
            color_tokens = catted_shape[:, -number_of_colors:]        # [bs, 313, C]
            new_patch_tokens[:, win] = catted_shape[:, :area]
        #################################################################
        return new_patch_tokens.view(batch, area, channal), color_tokens


class CustomNormalization(nn.Module):
    def __init__(self, normalization_layer, dim):
        super(CustomNormalization, self).__init__()
        self.normalizaiton_type = normalization_layer
        if normalization_layer == 'ln':
            self.norm = nn.LayerNorm(dim)
        elif normalization_layer == 'bn':
            self.norm = nn.BatchNorm1d(dim)
        elif normalization_layer == 'in':
            self.norm = nn.InstanceNorm1d(dim)
        elif normalization_layer == 'pn':
            self.norm = PixelNormalization(dim)
        # self.norm is the definition in torch _tensor

    def forward(self, custom_normalization):
        if self.normalizaiton_type == 'bn' or self.normalizaiton_type == 'in':
            custom_normalization = self.norm(custom_normalization.permute(0, 2, 1)).permute(0, 2, 1)
            return custom_normalization
        elif self.normalizaiton_type == 'none':
            return custom_normalization
        else:
            return self.norm(custom_normalization)


def window_partition(patch_token, window_size):
    # only for patch tokens.
    Batch, Height, Width, Channal = patch_token.shape
    patch_token = patch_token.view(Batch, Height // window_size, window_size, Width // window_size, window_size, Channal)
    windows = patch_token.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, Channal)
    # [B * H//w_size * W//w_size, W_size, W_size, C]
    return windows


def window_reverse(windows, window_size, Height, Width):
    # windows: (num_windows*B, window_size, window_size, C)->x: (B, H, W, C)
    Batch = int(windows.shape[0] / (Height * Width / window_size / window_size))
    viewed_window = windows.view(Batch, Height // window_size, Width // window_size, window_size, window_size, -1)
    viewed_window = viewed_window.permute(0, 1, 3, 2, 4, 5).contiguous().view(Batch, Height, Width, -1)
    return viewed_window


class Multiscale_Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, window_size, number_of_colors, num_windows, norm_layer='pn'):
        super(Multiscale_Block, self).__init__()
        self.window_size = window_size
        self.normalization1 = nn.LayerNorm(dim)
        self.normalization2 = nn.LayerNorm(dim)
        self.attention = Multidecoder_Attention(dim, heads, dropout, num_windows)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.multilayer_perceptron = FeedForward(dim, mlp_dim, dropout)
        self.number_of_windows = num_windows
        self.number_of_colors = number_of_colors          # 313

    def process_mask(self, mask, number_of_patch):
        # mask: [B, 256x256, 313] --> [B*num_windows, winsize*winsize+313, winsize*winsize+313]
        batch, number_of_pixel, number_of_colors = mask.size()
        original_height = original_width = int(np.sqrt(number_of_pixel))       # 256
        patch_height = patch_width = int(np.sqrt(number_of_patch))       # 16/32/64/128/256
        patch_size = original_height // patch_height       # 16/8/4/2/1
        mask = mask.view(batch, patch_height, patch_size, patch_width, patch_size, number_of_colors).permute(0, 1, 3, 2, 4, 5).contiguous()
        mask = mask.view(batch, patch_height, patch_width, patch_size*patch_size, number_of_colors)       # [B, p_h, p_w, psxps, 313]
        mask = torch.sum(mask, dim=-2)       # [B, p_h, p_w, 313]
        if self.number_of_windows == 1:
            mask = mask.view(batch, number_of_patch, number_of_colors)   # [B, N, 313]
            patch_ones = torch.ones(batch, number_of_patch, number_of_patch).cuda()
            color_ones = torch.ones(batch, number_of_colors, number_of_colors).cuda()
            process_mask_a = torch.cat((patch_ones, mask), dim=-1)      # [B, N, N+313]
            process_mask_b = torch.cat((mask.transpose(1, 2), color_ones), dim=-1)      # [B, 313, N+313]
            process_mask = torch.cat((process_mask_a, process_mask_b), dim=1)             # [B, N+313, N+313]
            # a and b are ab color
            return process_mask

        mask = window_partition(mask, self.window_size)     # [B * num_windows, win_size, win_size, 313]
        mask = mask.view(-1, self.window_size * self.window_size, number_of_colors)        # [B*num_windows, win-size * winsize, 313]
        number_of_window_tokens = self.window_size * self.window_size

        patch_ones = torch.ones(batch * self.number_of_windows, number_of_window_tokens, number_of_window_tokens).cuda()
        color_ones = torch.ones(batch * self.number_of_windows, number_of_colors, number_of_colors).cuda()
        process_mask_a = torch.cat((patch_ones, mask), dim=-1)              # [B*num, 16x16, 16x16+313]
        process_mask_b = torch.cat((mask.transpose(1, 2), color_ones), dim=-1)      # [B*num, 313, 16x16+313]
        process_mask = torch.cat((process_mask_a, process_mask_b), dim=1)       # [B*num_wins, 16*16+313, 16*16+313]
        # a and b are ab color
        return process_mask

    def forward(self, multiscale_block_neural, mask):
        batch, patch_tokens_plus_color_tokens, channal = multiscale_block_neural.size()      # N = patch_tokens + color_tokens
        number_of_patch = patch_tokens_plus_color_tokens - self.number_of_colors
        Height = Width = int(np.sqrt(number_of_patch))
        layer_normalization = self.normalization1(multiscale_block_neural)           # LayerNorm

        patch_tokens = layer_normalization[:, :-self.number_of_colors, :]        # [B, 16x16, C]
        if self.number_of_windows > 1:
            patch_tokens = patch_tokens.view(batch, Height, Width, channal)
            patch_tokens = window_partition(patch_tokens, self.window_size)
            patch_tokens = patch_tokens.view(-1, self.window_size * self.window_size, channal)    # [B * num_windows, w_size * w_size, C]

        color_tokens = layer_normalization[:, -self.number_of_colors:, :]        # [B, 313, C]
        attention_mask = self.process_mask(mask, number_of_patch)
        patch_tokens, color_tokens = self.attention(patch_tokens, color_tokens, attention_mask)

        if self.number_of_windows >1:
            patch_tokens = patch_tokens.view(-1, self.window_size, self.window_size, channal)
            patch_tokens = window_reverse(patch_tokens, self.window_size, Height, Width).view(batch, number_of_patch, channal)
        need_drop = torch.cat((patch_tokens, color_tokens), dim=1)
        multiscale_block_neural = multiscale_block_neural + self.drop_path(need_drop)
        multiscale_block_neural = multiscale_block_neural + self.drop_path(self.multilayer_perceptron(self.normalization2(multiscale_block_neural)))
        del attention_mask

        return multiscale_block_neural



######### use color tokens as condition ######################################
class Decoder_Block_Color(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.normalization1 = nn.LayerNorm(dim)
        self.normalization2 = nn.LayerNorm(dim)
        self.heads = heads
        self.attention = MultiheadAttention(dim, heads, dropout)
        self.multilayer_perceptron = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, decode_color_token_neural, color_embed, mask=None):
        decode_color_token_neural = self.normalization1(decode_color_token_neural)
        tokens = torch.cat((decode_color_token_neural, color_embed), dim=1)
        need_drop = self.attention(query=decode_color_token_neural,
                                   key=tokens,
                                   value=tokens,
                                   attn_mask=mask)[0]
        decode_color_token_neural = decode_color_token_neural + self.drop_path(need_drop)
        decode_color_token_neural = decode_color_token_neural + self.drop_path(self.multilayer_perceptron(self.normalization2(decode_color_token_neural)))
        return decode_color_token_neural


class MultiheadAttention(nn.Module):         # for vit attention
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dimension = dimension // heads
        self.scale = head_dimension ** -0.5
        self.attention = None

        self.get_query = nn.Linear(dimension, dimension)
        self.get_key = nn.Linear(dimension, dimension)
        self.get_value = nn.Linear(dimension, dimension)

        self.attention_drop = nn.Dropout(dropout)
        self.project = nn.Linear(dimension, dimension)
        self.project_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, query, key, value, attention_mask=None):
        batch, area, channal = query.shape
        luminance = key.shape[1]

        query = self.get_query(query).reshape(batch, area, self.heads, channal // self.heads).permute(0, 2, 1, 3)
        key = self.get_key(key).reshape(batch, luminance, self.heads, channal // self.heads).permute(0, 2, 1, 3)
        value = self.get_value(value).reshape(batch, luminance, self.heads, channal // self.heads).permute(0, 2, 1, 3)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            expand_mask = attention_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)     # [B, heads, N, L]
            attention = attention.masked_fill(expand_mask == 0, -float('inf'))
        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention)
        out = (attention @ value).transpose(1, 2).reshape(batch, area, channal)
        out = self.project(out)
        out = self.project_drop(out)

        return out, attention
