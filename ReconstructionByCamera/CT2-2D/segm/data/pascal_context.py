from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_directory

PASCAL_CONTEXT_CONFIG_PATH = Path(__file__).parent / "config" / "pascal_context.py"
PASCAL_CONTEXT_CATS_PATH = Path(__file__).parent / "config" / "pascal_context.yml"


class PascalContextDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size, crop_size, split, PASCAL_CONTEXT_CONFIG_PATH, **kwargs
        )
        self.names, self.colors = utils.dataset_cat_description(
            PASCAL_CONTEXT_CATS_PATH
        )
        self.number_of_colors = 60
        self.ignore_label = 255
        self.reduce_zero_label = False

    def update_default_config(self, config):
        root_directory = dataset_directory()
        path = Path(root_directory) / "pcontext"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path / "VOCdevkit/VOC2010/"
        elif self.split == "val":
            config.data.val.data_root = path / "VOCdevkit/VOC2010/"
        elif self.split == "test":
            raise ValueError("Test split is not valid for Pascal Context dataset")
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels
