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


class COCODataset(Dataset):
    def __init__(
        self,
        image_size=224,
        crop_size=224,
        split="train",
        normalization="vit",
        dataset_directory='',
        add_mask=False,
        patch_size=16,
        change_mask=False,
        multi_scaled=False,
        mask_number=4,
        mask_random=False,
        number_of_colors=313,
    ):
        super().__init__()
        self.dataset_directory = dataset_directory
        self.image_directory = os.path.join(self.dataset_directory, split)       # for imagenet
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
            assert os.path.exists(os.path.join(self.dataset_directory, 'mask_prior.pickle'))
            file_open = open(os.path.join(self.dataset_directory, 'mask_prior.pickle'), 'rb')
            load_dictionary = pickle.load(file_open)
            self.mask_luminance = np.zeros((mask_number, 313)).astype(np.bool)     # [4, 313]
            for key in range(101):
                for mask_number in range(mask_number):
                    start_key = mask_number * (100 // mask_number)      # 0
                    end_key = (mask_number+1)* (100 // mask_number)     # 25
                    if start_key <= key < end_key:
                        self.mask_luminance[mask_number, :] += load_dictionary[key].astype(np.bool)
                        break
            # number_of_colors
            if self.number_of_colors != 313:
                squeeze_mask = np.zeros((mask_number, self.number_of_colors)).astype(np.bool)     # [4, 78]
                multiple = 313 // self.number_of_colors
                for mask_number in range(mask_number):
                    for index in range(self.number_of_colors):
                        dimension_sum = np.sum(self.mask_luminance[mask_number, index * multiple: (index + 1) * multiple - 1], axis=-1)
                        squeeze_mask[mask_number, index] += dimension_sum
                self.mask_luminance = squeeze_mask

            self.mask_luminance = self.mask_luminance.astype(np.float32)
            self.random_mask_luminance = np.zeros_like(self.mask_luminance)

    @property
    def unwrapped(self):
        return self

    def load_filenames(self, data_directory, split, filepath='fullfilenames.pickle'):
        if split == 'train':
            split_filepth = os.path.join(data_directory, 'clean_train_filenames.pickle')
        else:
            split_filepth = os.path.join(data_directory, split + '_' + filepath)
        if not os.path.exists(split_filepth):
            if split == 'train':
                classnames = os.listdir(self.image_directory)
                filenames = []
                for class_element in classnames:
                    class_filenames = os.listdir(os.path.join(self.image_directory, class_element))
                    for ele in class_filenames:
                        filenames.append([class_element, ele])
                local_split_filepath = os.path.join(data_directory, split + '_' + filepath)
                file_open = open(local_split_filepath, 'wb')
                pickle.dump(filenames, file_open)
                print('save to:', local_split_filepath)
                file_open.close()
            elif split == 'val':
                local_split_filepath = os.path.join(data_directory, split + '_' + filepath)
                filenames = []
                for validation_index in range(1, 5001):
                    file_name = 'ILSVRC2012_val_' + str(validation_index).zfill(8) + '.JPEG'
                    filenames.append(file_name)
                file_open = open(local_split_filepath, 'wb')
                pickle.dump(filenames, file_open)
                print('save to:', local_split_filepath)
                file_open.close()
        else:
            file_open = open(split_filepth, 'rb')
            filenames = pickle.load(file_open)
            print('Load from:', split_filepth)
        return filenames


    def rgb_to_lab(self, image):
        assert image.dtype == np.uint8
        return color.rgb2lab(image).astype(np.float32)

    def numpy_to_torch(self, image):
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0))      # [c, h, w]
        return tensor.type(torch.float32)

    def get_img(self, key):
        if self.split == 'train':
            assert len(key) == 2
            image_path = os.path.join(self.image_directory, key[0], key[1])
        else:
            image_path = os.path.join(self.image_directory, key)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if self.split == 'train':
            image_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            # mini_size = min(width, height)
            # image_transform = transforms.Compose([
            #     transforms.CenterCrop(mini_size),
            #     transforms.Resize(self.crop_size),
            # ])
            image_transform = transforms.Compose([
                transforms.Resize(self.crop_size),
            ])
        image_resized = image_transform(image)
        image_resized = np.array(image_resized)

        luminance_resized = self.rgb_to_lab(image_resized)[:, :, :1]
        ab_resized = self.rgb_to_lab(image_resized)[:, :, 1:]     # np.float32
        mask = torch.ones(1)
        if self.add_mask:
            original_lab = luminance_resized[:, :, 0]
            lab = original_lab.reshape((self.crop_size * self.crop_size))
            mask_patch_color = np.zeros((self.crop_size**2, self.number_of_colors), dtype=np.float32)

            for luminance_range in range(self.mask_number):
                start_l1, end_l1 = luminance_range * (100 // self.mask_number), (luminance_range + 1) * (100 // self.mask_number)
                if end_l1 == 100:
                    index_l1 = np.where((lab >= start_l1) & (lab <= end_l1))[0]
                else:
                    index_l1 = np.where((lab >= start_l1) & (lab < end_l1))[0]

                if not self.mask_random:
                    mask_patch_color[index_l1, :] = self.mask_luminance[luminance_range, :]
                else:
                    mask_patch_color[index_l1, :] = self.random_mask_luminance[luminance_range, :]

            mask = torch.from_numpy(mask_patch_color)
        image_luminance = self.numpy_to_torch(luminance_resized)
        image_ab = self.numpy_to_torch(ab_resized)

        return image_luminance, image_ab, mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        key = self.filenames[index]
        image_luminance, image_ab, mask = self.get_img(key)
        return image_luminance, image_ab, key, mask

