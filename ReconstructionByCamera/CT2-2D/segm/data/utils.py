import torch
import torchvision.transforms.functional as torch_functional
import numpy as np
import yaml
from pathlib import Path

IGNORE_LABEL = 255
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def segmentation_to_rgb(segmentation, colors):
    image = torch.zeros((segmentation.shape[0], segmentation.shape[1], segmentation.shape[2], 3)).float()
    colors = torch.unique(segmentation)
    for color_ in colors:
        color = colors[int(color_)]
        if len(color.shape) > 1:
            color = color[0]
        image[segmentation == color_] = color
    return image


def dataset_cat_description(path, color_map=None):
    desc = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    colors = {}
    names = []
    for i, cat in enumerate(desc):
        names.append(cat["name"])
        if "color" in cat:
            colors[cat["id"]] = torch.tensor(cat["color"]).float() / 255
        else:
            colors[cat["id"]] = torch.tensor(color_map[cat["id"]]).float()
    colors[IGNORE_LABEL] = torch.tensor([0.0, 0.0, 0.0]).float()
    return names, colors


def rgb_normalize(normalization, stats):
    """
    x : C x *
    x \in [0, 1]
    """
    return torch_functional.normalize(normalization, stats["mean"], stats["std"])


def rgb_denormalize(demornalization, stats):
    """
    x : N x C x *
    x \in [-1, 1]
    """
    mean = torch.tensor(stats["mean"])
    standard_deviation = torch.tensor(stats["std"])
    for i in range(3):
        demornalization[:, i, :, :] = demornalization[:, i, :, :] * standard_deviation[i] + mean[i]
    return demornalization
