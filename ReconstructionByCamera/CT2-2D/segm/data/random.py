import os
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from skimage.io import imread
# from skimage import transform, color
from skimage import color

from segm.data import utils
import pickle
import torch
from segm.config import dataset_directory
# import cv2
import collections
import json


class RandomDataset(Dataset):
    def __init__(
            self,
            image_size=256,
            crop_size=256,
            split="val",
            normalization="vit",
            dataset_directory='',
            add_mask=False,
            patch_size=16,
            change_mask=False,
            multi_scaled=False,
            mask_number=4,
            mask_random=False,
            number_of_colors = 313
    ):
        super().__init__()
        self.dataset_directory = dataset_directory
        self.crop_size = crop_size
        self.image_size = image_size
        self.split = split
        self.add_mask = add_mask
        self.patch_size = patch_size
        self.change_mask = change_mask
        self.multi_scaled = multi_scaled
        self.mask_number = mask_number
        self.mask_random = mask_random
        assert self.crop_size % self.patch_size == 0

        self.filenames = self.load_filenames(self.dataset_directory, split)
        self.number_of_colors = number_of_colors
        if self.add_mask:
            assert os.path.exists(
                os.path.join(self.dataset_directory, 'mask_prior.pickle'))
            file_open = open(os.path.join(self.dataset_directory, 'mask_prior.pickle'), 'rb')
            load_dictionary = pickle.load(file_open)

            self.mask_luminance = np.zeros((mask_number, 313)).astype(np.bool)  # [4, 313]
            for key in range(101):
                for mask_number_ in range(mask_number):
                    start_key = mask_number_ * (100 // mask_number)  # 0
                    end_key = (mask_number_ + 1) * (100 // mask_number)  # 25
                    if start_key <= key < end_key:
                        self.mask_luminance[mask_number_, :] += load_dictionary[key].astype(np.bool)
                        break

            self.mask_luminance = self.mask_luminance.astype(np.float32)
            del load_dictionary

    @property
    def unwrapped(self):
        return self

    def load_filenames(self, data_dir, split, filepath='fullfilenames.pickle'):
        filenames = os.listdir(data_dir)
        return filenames

    def rgb_to_lab(self, image):
        assert image.dtype == np.uint8
        return color.rgb2lab(image).astype(np.float32)

    def numpy_to_torch(self, image):
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0))  # [c, h, w]
        return tensor.type(torch.float32)

    def get_image(self, key):
        image_path = os.path.join(self.dataset_directory, key)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # if width != 256 or height != 256:
        #     mini_size = min(width, height)
        #     image_transform = transforms.Compose([
        #         transforms.CenterCrop(mini_size),
        #         transforms.Resize(self.crop_size),
        #     ])
        #     image = image_transform(image)
        image_transform = transforms.Compose([
            transforms.Resize(self.crop_size),
        ])
        image = image_transform(image)

        image_resized = np.array(image)
        luminance_resized = self.rgb_to_lab(image_resized)[:, :, :1]
        ab_resized = self.rgb_to_lab(image_resized)[:, :, 1:]  # np.float32

        mask = torch.ones(1)
        if self.add_mask:
            original_lab = luminance_resized[:, :, 0]
            lab = original_lab.reshape((self.crop_size * self.crop_size))
            mask_patch_color = np.zeros((self.crop_size ** 2, self.number_of_colors),
                                dtype=np.float32)

            for luminance_range in range(self.mask_number):
                start_l1, end_l1 = luminance_range * (100 // self.mask_number), (
                            luminance_range + 1) * (100 // self.mask_number)
                if end_l1 == 100:
                    index_l1 = np.where((lab >= start_l1) & (lab <= end_l1))[0]
                else:
                    index_l1 = np.where((lab >= start_l1) & (lab < end_l1))[0]
                mask_patch_color[index_l1, :] = self.mask_luminance[luminance_range, :]

            mask = torch.from_numpy(mask_patch_color)
        image_luminance = self.numpy_to_torch(luminance_resized)
        image_ab = self.numpy_to_torch(ab_resized)

        return image_luminance, image_ab, mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        key = self.filenames[index]
        image_luminance, image_ab, mask = self.get_image(key)
        return image_luminance, image_ab, key, mask
