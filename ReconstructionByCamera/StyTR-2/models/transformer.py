import copy
from typing import Optional, List

import torch
import torch.nn.functional as torch_functional
from torch import Tensor
import torch.nn as torch_neural
from function import normal, normal_style
import numpy as np
import os

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


class Transformer(torch_neural.Module):

    def __init__(self, dimension_model=512, number_of_heads=8, number_encoder_layers=3,
                 number_of_decoder_layers=3, dimension_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_decoder=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(dimension_model, number_of_heads, dimension_feedforward,
                                                dropout, activation,
                                                normalize_before)
        encoder_normalization = torch_neural.LayerNorm(dimension_model) if normalize_before else None
        self.encoder_content = TransformerEncoder(encoder_layer, number_encoder_layers,
                                                  encoder_normalization)
        self.encoder_style = TransformerEncoder(encoder_layer, number_encoder_layers,
                                                encoder_normalization)

        decoder_layer = TransformerDecoderLayer(dimension_model, number_of_heads, dimension_feedforward,
                                                dropout, activation,
                                                normalize_before)
        decoder_normalization = torch_neural.LayerNorm(dimension_model)
        self.decoder = TransformerDecoder(decoder_layer, number_of_decoder_layers,
                                          decoder_normalization,
                                          return_intermediate=return_intermediate_decoder)

        self._reset_parameters()

        self.dimension_model = dimension_model
        self.number_of_head = number_of_heads

        self.new_positions = torch_neural.Conv2d(512, 512, (1, 1))
        self.averagepooling = torch_neural.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for parameter_ in self.parameters():
            if parameter_.dim() > 1:
                torch_neural.init.xavier_uniform_(parameter_)

    def forward(self, style, mask, content, position_embed_content, position_embed_style):

        # content-aware positional embedding
        content_pool = self.averagepooling(content)
        position_content = self.new_positions(content_pool)
        # pos_embed_c = F.interpolate(pos_c, mode='bilinear',
        #                             size=style.shape[-2:])
        position_embed_content = torch_functional.interpolate(position_content, mode='bilinear', align_corners=True,
                                                              size=style.shape[-2:])


        ###flatten NxCxHxW to HWxNxC     
        style = style.flatten(2).permute(2, 0, 1)
        if position_embed_style is not None:
            position_embed_style = position_embed_style.flatten(2).permute(2, 0, 1)

        content = content.flatten(2).permute(2, 0, 1)
        if position_embed_content is not None:
            position_embed_content = position_embed_content.flatten(2).permute(2, 0, 1)

        style = self.encoder_style(style, source_key_padding_mask=mask,
                                   position=position_embed_style)
        content = self.encoder_content(content, source_key_padding_mask=mask,
                                       position=position_embed_content)
        decoded_style = self.decoder(content, style, memory_key_padding_mask=mask,
                                     position=position_embed_style, query_position=position_embed_content)[0]

        ### HWxNxC to NxCxHxW to
        area, batch, channal = decoded_style.shape
        height = int(np.sqrt(area))
        decoded_style = decoded_style.permute(1, 2, 0)
        decoded_style = decoded_style.view(batch, channal, -1, height)

        return decoded_style


class TransformerEncoder(torch_neural.Module):

    def __init__(self, encoder_layer, number_of_layers, normalization=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, number_of_layers)
        self.number_of_layers = number_of_layers
        self.normalization = normalization

    def forward(self, source,
                mask: Optional[Tensor] = None,
                source_key_padding_mask: Optional[Tensor] = None,
                position: Optional[Tensor] = None):
        output = source

        for layer in self.layers:
            output = layer(output, source_mask=mask,
                           source_key_padding_mask=source_key_padding_mask, position=position)

        if self.normalization is not None:
            output = self.normalization(output)

        return output


class TransformerDecoder(torch_neural.Module):

    def __init__(self, decoder_layer, number_of_layers, normalization=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, number_of_layers)
        self.number_of_layers = number_of_layers
        self.normalization = normalization
        self.return_intermediate = return_intermediate

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                position: Optional[Tensor] = None,
                query_position: Optional[Tensor] = None):
        output = target

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, target_mask=target_mask,
                           memory_mask=memory_mask,
                           target_key_padding_mask=target_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           position=position, query_position=query_position)
            if self.return_intermediate:
                intermediate.append(self.normalization(output))

        if self.normalization is not None:
            output = self.normalization(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(torch_neural.Module):

    def __init__(self, dimension_model, number_of_head, dimension_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attention = torch_neural.MultiheadAttention(dimension_model, number_of_head, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch_neural.Linear(dimension_model, dimension_feedforward)
        self.dropout = torch_neural.Dropout(dropout)
        self.linear2 = torch_neural.Linear(dimension_feedforward, dimension_model)

        self.normalization1 = torch_neural.LayerNorm(dimension_model)
        self.normalization2 = torch_neural.LayerNorm(dimension_model)
        self.dropout1 = torch_neural.Dropout(dropout)
        self.dropout2 = torch_neural.Dropout(dropout)

        self.activation = _get_activation_function(activation)
        self.normalize_before = normalize_before

    def with_position_embed(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position

    def forward_position(self,
                         source,
                         source_mask: Optional[Tensor] = None,
                         source_key_padding_mask: Optional[Tensor] = None,
                         position: Optional[Tensor] = None):
        query = key = self.with_position_embed(source, position)
        # q = k = src
        # print(q.size(),k.size(),src.size())
        source2 = self.self_attention(query, key, value=source, attn_mask=source_mask,
                                      key_padding_mask=source_key_padding_mask)[0]
        # attn_mask in torch_neural.MultiheadAttention
        source = source + self.dropout1(source2)
        source = self.normalization1(source)
        source2 = self.linear2(self.dropout(self.activation(self.linear1(source))))
        source = source + self.dropout2(source2)
        source = self.normalization2(source)
        return source

    def forward_prediction(self, source,
                           source_mask: Optional[Tensor] = None,
                           source_key_padding_mask: Optional[Tensor] = None,
                           position: Optional[Tensor] = None):
        source2 = self.normalization1(source)
        query = key = self.with_position_embed(source2, position)
        source2 = self.self_attention(query, key, value=source2, attention_mask=source_mask,
                                      key_padding_mask=source_key_padding_mask)[0]
        source = source + self.dropout1(source2)
        source2 = self.normalization2(source)
        source2 = self.linear2(self.dropout(self.activation(self.linear1(source2))))
        source = source + self.dropout2(source2)
        return source

    def forward(self, source,
                source_mask: Optional[Tensor] = None,
                source_key_padding_mask: Optional[Tensor] = None,
                position: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_prediction(source, source_mask, source_key_padding_mask, position)
        return self.forward_position(source, source_mask, source_key_padding_mask, position)


class TransformerDecoderLayer(torch_neural.Module):

    def __init__(self, dimension_model, number_of_heads, dimension_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # d_model embedding dim
        self.self_attention = torch_neural.MultiheadAttention(dimension_model, number_of_heads, dropout=dropout)
        self.multihead_attention = torch_neural.MultiheadAttention(dimension_model, number_of_heads,
                                                                   dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch_neural.Linear(dimension_model, dimension_feedforward)
        self.dropout = torch_neural.Dropout(dropout)
        self.linear2 = torch_neural.Linear(dimension_feedforward, dimension_model)

        self.normalization1 = torch_neural.LayerNorm(dimension_model)
        self.normalization2 = torch_neural.LayerNorm(dimension_model)
        self.normalization3 = torch_neural.LayerNorm(dimension_model)
        self.dropout1 = torch_neural.Dropout(dropout)
        self.dropout2 = torch_neural.Dropout(dropout)
        self.dropout3 = torch_neural.Dropout(dropout)

        self.activation = _get_activation_function(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position

    def forward_position(self, target, memory,
                         target_mask: Optional[Tensor] = None,
                         memory_mask: Optional[Tensor] = None,
                         target_key_padding_mask: Optional[Tensor] = None,
                         memory_key_padding_mask: Optional[Tensor] = None,
                         position: Optional[Tensor] = None,
                         query_position: Optional[Tensor] = None):
        query = self.with_pos_embed(target, query_position)
        key = self.with_pos_embed(memory, position)
        value = memory

        target2 = self.self_attention(query, key, value, attn_mask=target_mask,
                                   key_padding_mask=target_key_padding_mask)[0]

        target = target + self.dropout1(target2)
        target = self.normalization1(target)
        target2 = self.multihead_attention(query=self.with_pos_embed(target, query_position),
                                        key=self.with_pos_embed(memory, position),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        target = target + self.dropout2(target2)
        target = self.normalization2(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = target + self.dropout3(target2)
        target = self.normalization3(target)
        return target

    def forward_prediction(self, target, memory,
                           target_mask: Optional[Tensor] = None,
                           memory_mask: Optional[Tensor] = None,
                           target_key_padding_mask: Optional[Tensor] = None,
                           memory_key_padding_mask: Optional[Tensor] = None,
                           position: Optional[Tensor] = None,
                           query_position: Optional[Tensor] = None):
        target2 = self.normalization1(target)
        query = key = self.with_pos_embed(target2, query_position)
        target2 = self.self_attention(query, key, value=target2, attn_mask=target_mask,
                                   key_padding_mask=target_key_padding_mask)[0]

        target = target + self.dropout1(target2)
        target2 = self.normalization2(target)
        target2 = self.multihead_attention(query=self.with_pos_embed(target2, query_position),
                                        key=self.with_pos_embed(memory, position),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]

        target = target + self.dropout2(target2)
        target2 = self.normalization3(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout3(target2)
        return target

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                position: Optional[Tensor] = None,
                query_position: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_prediction(target, memory, target_mask, memory_mask,
                                           target_key_padding_mask,
                                           memory_key_padding_mask, position, query_position)
        return self.forward_position(target, memory, target_mask, memory_mask,
                                     target_key_padding_mask, memory_key_padding_mask,
                                     position, query_position)


def _get_clones(module, number):
    return torch_neural.ModuleList([copy.deepcopy(module) for i in range(number)])


def build_transformer(args):
    return Transformer(
        dimension_model=args.hidden_dimension,
        dropout=args.dropout,
        number_of_heads=args.number_of_heads,
        dimension_feedforward=args.dimension_feedforward,
        number_encoder_layers=args.encoder_layers,
        number_of_decoder_layers=args.decoder_layers,
        normalize_before=args.prediction_normalization,
        return_intermediate_decoder=True,
    )


def _get_activation_function(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch_functional.relu
    if activation == "gelu":
        return torch_functional.gelu
    if activation == "glu":
        return torch_functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
