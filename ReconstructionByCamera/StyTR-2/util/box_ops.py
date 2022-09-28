# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(input):
    x_c, y_c, width, height = input.unbind(-1)
    box = [(x_c - 0.5 * width), (y_c - 0.5 * height),
           (x_c + 0.5 * width), (y_c + 0.5 * height)]
    return torch.stack(box, dim=-1)


def box_xyxy_to_cxcywh(input):
    x0, y0, x1, y1 = input.unbind(-1)
    box = [(x0 + x1) / 2, (y0 + y1) / 2,
           (x1 - x0), (y1 - y0)]
    return torch.stack(box, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    max_box = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    min_box = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    differences_positive_only = (min_box - max_box).clamp(min=0)  # [N,M,2]
    inter = differences_positive_only[:, :, 0] * differences_positive_only[:, :,
                                                 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    min_box = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    max_box = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    differences_positive_only = (max_box - min_box).clamp(min=0)  # [N,M,2]
    area = differences_positive_only[:, :, 0] * differences_positive_only[:, :,
                                                1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    height, width = masks.shape[-2:]

    aranged_height = torch.arange(0, height, dtype=torch.float)
    aranged_width = torch.arange(0, width, dtype=torch.float)
    aranged_height, aranged_width = torch.meshgrid(aranged_height,
                                                   aranged_width)

    aranged_width_mask = (masks * aranged_width.unsqueeze(0))
    aranged_width_max = aranged_width_mask.flatten(1).max(-1)[0]
    aranged_width_min = \
    aranged_width_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    aranged_height_mask = (masks * aranged_height.unsqueeze(0))
    aranged_height_max = aranged_height_mask.flatten(1).max(-1)[0]
    aranged_height_min = \
    aranged_height_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack(
        [aranged_width_min, aranged_height_min, aranged_width_max,
         aranged_height_max], 1)
