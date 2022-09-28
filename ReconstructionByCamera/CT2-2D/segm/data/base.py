import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as torch_functional

from mmseg.datasets import build_dataset
import mmcv
from mmcv.utils import Config


from segm.data.utils import STATS, IGNORE_LABEL
from segm.data import utils


class BaseMMSeg(Dataset):
    def __init__(
        self,
        image_size,
        crop_size,
        split,
        config_path,
        normalization,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.normalization = STATS[normalization].copy()
        self.ignore_label = None
        for key_, value_ in self.normalization.items():
            value_ = np.round(255 * np.array(value_), 2)
            self.normalization[key_] = tuple(value_)
        print(f"Use normalization: {self.normalization}")

        config = Config.fromfile(config_path)

        self.ratio = config.max_ratio
        self.dataset = None
        self.config = self.update_default_config(config)
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))

    def update_default_config(self, config):

        train_splits = ["train", "trainval"]
        if self.split in train_splits:
            config_pipeline = getattr(config, f"train_pipeline")
        else:
            config_pipeline = getattr(config, f"{self.split}_pipeline")

        image_scale = (self.ratio * self.image_size, self.image_size)
        if self.split not in train_splits:
            assert config_pipeline[1]["type"] == "MultiScaleFlipAug"
            config_pipeline = config_pipeline[1]["transforms"]
        for i, option in enumerate(config_pipeline):
            option_type = option["type"]
            if option_type == "Resize":
                option["img_scale"] = image_scale
            elif option_type == "RandomCrop":
                option["crop_size"] = (
                    self.crop_size,
                    self.crop_size,
                )
            elif option_type == "Normalize":
                option["mean"] = self.normalization["mean"]
                option["std"] = self.normalization["std"]
            elif option_type == "Pad":
                option["size"] = (self.crop_size, self.crop_size)
            config_pipeline[i] = option
        if self.split == "train":
            config.data.train.pipeline = config_pipeline
        elif self.split == "trainval":
            config.data.trainval.pipeline = config_pipeline
        elif self.split == "val":
            config.data.val.pipeline[1]["img_scale"] = image_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
        elif self.split == "test":
            config.data.test.pipeline[1]["img_scale"] = image_scale
            config.data.test.pipeline[1]["transforms"] = config_pipeline
            config.data.test.test_mode = True
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return config

    def set_multiscale_mode(self):
        self.config.data.val.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.val.pipeline[1]["flip"] = True
        self.config.data.test.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.test.pipeline[1]["flip"] = True
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))

    def __getitem__(self, index):
        data = self.dataset[index]

        train_splits = ["train", "trainval"]

        if self.split in train_splits:
            image = data["image"].data
            segmentation = data["gt_semantic_seg"].data.squeeze(0)
        else:
            image = [image.data for image in data["image"]]
            segmentation = None

        out = dict(im=image)
        if self.split in train_splits:
            out["segmentation"] = segmentation
        else:
            image_metas = [meta.data for meta in data["img_metas"]]
            out["im_metas"] = image_metas
            out["colors"] = self.colors

        return out

    def get_ground_truth_segmentation_maps(self):
        dataset = self.dataset
        ground_truth_segmentation_maps = {}
        for image_info in dataset.img_infos:
            seg_map = Path(dataset.ann_dir) / image_info["ann"]["seg_map"]
            ground_truth_segmentation_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
            ground_truth_segmentation_map[ground_truth_segmentation_map == self.ignore_label] = IGNORE_LABEL
            if self.reduce_zero_label:
                ground_truth_segmentation_map[ground_truth_segmentation_map != IGNORE_LABEL] -= 1
            ground_truth_segmentation_maps[image_info["filename"]] = ground_truth_segmentation_map
        return ground_truth_segmentation_maps

    def __len__(self):
        return len(self.dataset)

    @property
    def unwrapped(self):
        return self

    def set_epoch(self, epoch):
        pass

    def get_diagnostics(self, logger):
        pass

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return
