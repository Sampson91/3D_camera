from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from segm.model.vit import VisionTransformer
from segm.model.utils import checkpoint_filter_function
from segm.model.decoder import DecoderLinear
from segm.model.decoder import MaskTransformer
from segm.model.segmenter import Segmenter
import segm.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model




def create_vit(model_config):
    model_config = model_config.copy()
    backbone = model_config.pop("backbone")        # vit_tiny_patch16_384

    normalization = model_config.pop("normalization")
    model_config["number_of_colors"] = 1000
    mlp_expansion_ratio = 4
    model_config["dimension_feedforward"] = mlp_expansion_ratio * model_config["number_of_features"]

    if backbone in default_cfgs:
        print('backbone in timm.visual_t.default_cfgs', backbone)
        default_config = default_cfgs[backbone]
    else:
        default_config = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_config["input_size"] = (
        3,
        model_config["image_size"][0],
        model_config["image_size"][1],
    )
    model = VisionTransformer(**model_config)

    if backbone == 'vit_tiny_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_path = os.path.join(os.getcwd(), 'segm/resources/vit_tiny_patch16_384.npz')
        model.load_pretrained(pretrained_vit_path)

    elif backbone == 'vit_small_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_path = os.path.join(os.getcwd(), 'segm/resources/vit_small_patch16_384.npz')
        model.load_pretrained(pretrained_vit_path)

    elif backbone == 'vit_base_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_path = os.path.join(os.getcwd(), 'segm/resources/vit_base_patch16_384.npz')
        model.load_pretrained(pretrained_vit_path)

    elif backbone == 'vit_large_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_path = os.path.join(os.getcwd(), 'segm/resources/vit_large_patch16_384.npz')
        model.load_pretrained(pretrained_vit_path)
    elif "deit" in backbone:
        load_pretrained(model, default_config, filter_fn=checkpoint_filter_function)
    else:
        pretrained_vit_path = os.path.join(os.getcwd(), 'segm/resources/vit_tiny_patch16_224.npz')
        model.load_pretrained(pretrained_vit_path)

    return model


def create_decoder(encoder, decoder_config, backbone):
    decoder_config = decoder_config.copy()
    name = decoder_config.pop("name")
    if 'vit' in backbone:
        decoder_config["decoder_encoder"] = encoder.number_of_features      # 192 for vit-tiny
    decoder_config["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_config)
    elif name == "mask_transformer":
        if 'vit' in backbone:
            dimension = encoder.number_of_features
            number_of_heads = dimension // 64     # original_dim = 192
        decoder_config["number_of_heads"] = number_of_heads
        decoder_config["number_of_features"] = dimension
        decoder_config["dimension_feedforward"] = 4 * dimension
        decoder = MaskTransformer(**decoder_config)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_config):
    model_config = model_config.copy()
    decoder_config = model_config.pop("decoder")
    decoder_config["number_of_colors"] = model_config["number_of_colors"]

    if 'vit' in model_config['backbone']:
        encoder = create_vit(model_config)

    decoder = create_decoder(encoder, decoder_config, model_config['backbone'])
    model = Segmenter(encoder, decoder, number_of_colors=model_config["number_of_colors"], before_classify=decoder_config['before_classify'], backbone=model_config['backbone'], without_classification=decoder_config['without_classification'])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as file:
        variant = yaml.load(file, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
