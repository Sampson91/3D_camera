import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings

from option import args

warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        random_randint_1 = np.random.randint(0, 4)
        sample['low_resolution_image'] = np.rot90(
            sample['low_resolution_image'], random_randint_1).copy()
        sample['high_resolution'] = np.rot90(sample['high_resolution'],
                                             random_randint_1).copy()
        sample['upsampled_low_resolution_image'] = np.rot90(
            sample['upsampled_low_resolution_image'], random_randint_1).copy()
        random_randint_2 = np.random.randint(0, 4)
        sample['high_resolution_images_as_references'] = np.rot90(
            sample['high_resolution_images_as_references'],
            random_randint_2).copy()
        sample['down_and_upsampled_Ref_image'] = np.rot90(
            sample['down_and_upsampled_Ref_image'], random_randint_2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['low_resolution_image'] = np.fliplr(
                sample['low_resolution_image']).copy()
            sample['high_resolution'] = np.fliplr(
                sample['high_resolution']).copy()
            sample['upsampled_low_resolution_image'] = np.fliplr(
                sample['upsampled_low_resolution_image']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['high_resolution_images_as_references'] = np.fliplr(
                sample['high_resolution_images_as_references']).copy()
            sample['down_and_upsampled_Ref_image'] = np.fliplr(
                sample['down_and_upsampled_Ref_image']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['low_resolution_image'] = np.flipud(
                sample['low_resolution_image']).copy()
            sample['high_resolution'] = np.flipud(
                sample['high_resolution']).copy()
            sample['upsampled_low_resolution_image'] = np.flipud(
                sample['upsampled_low_resolution_image']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['high_resolution_images_as_references'] = np.flipud(
                sample['high_resolution_images_as_references']).copy()
            sample['down_and_upsampled_Ref_image'] = np.flipud(
                sample['down_and_upsampled_Ref_image']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        low_resolution_image, upsampled_low_resolution_image, high_resolution, high_resolution_images_as_references, down_and_upsampled_Ref_image = \
            sample['low_resolution_image'], sample[
                'upsampled_low_resolution_image'], sample['high_resolution'], \
            sample['high_resolution_images_as_references'], sample[
                'down_and_upsampled_Ref_image']
        low_resolution_image = low_resolution_image.transpose((2, 0, 1))
        upsampled_low_resolution_image = upsampled_low_resolution_image.transpose(
            (2, 0, 1))
        high_resolution = high_resolution.transpose((2, 0, 1))
        high_resolution_images_as_references = high_resolution_images_as_references.transpose(
            (2, 0, 1))
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.transpose(
            (2, 0, 1))
        return {
            'low_resolution_image': torch.from_numpy(
                low_resolution_image).float(),
            'upsampled_low_resolution_image': torch.from_numpy(
                upsampled_low_resolution_image).float(),
            'high_resolution': torch.from_numpy(high_resolution).float(),
            'high_resolution_images_as_references': torch.from_numpy(
                high_resolution_images_as_references).float(),
            'down_and_upsampled_Ref_image': torch.from_numpy(
                down_and_upsampled_Ref_image).float()
        }


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose(
        [RandomFlip(), RandomRotate(), ToTensor()])):
        self.input_list = sorted(
            [os.path.join(args.dataset_dir, 'train/input', name) for name in
             os.listdir(os.path.join(args.dataset_dir, 'train/input'))])
        self.high_resolution_images_as_references_list = sorted(
            [os.path.join(args.dataset_dir, 'train/ref', name) for name in
             os.listdir(os.path.join(args.dataset_dir, 'train/ref'))])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        high_resolution = imread(self.input_list[idx])
        height, width = high_resolution.shape[:2]
        #high_resolution = high_resolution[:height // 4 * 4, :width // 4 * 4, :]

        ### LR and LR_sr
        low_resolution_image = np.array(
            Image.fromarray(high_resolution).resize((width // 4, height // 4),
                                                    Image.BICUBIC))
        upsampled_low_resolution_image = np.array(
            Image.fromarray(low_resolution_image).resize((width, height),
                                                         Image.BICUBIC))

        ### Ref and Ref_sr
        high_resolution_images_as_references_sub = imread(
            self.high_resolution_images_as_references_list[idx])
        height_2, width_2 = high_resolution_images_as_references_sub.shape[:2]
        down_and_upsampled_Ref_image_sub = np.array(
            Image.fromarray(high_resolution_images_as_references_sub).resize(
                (width_2 // 4, height_2 // 4), Image.BICUBIC))
        down_and_upsampled_Ref_image_sub = np.array(
            Image.fromarray(down_and_upsampled_Ref_image_sub).resize(
                (width_2, height_2), Image.BICUBIC))

        ### complete ref and ref_sr to the same size, to use batch_size > 1
        high_resolution_images_as_references = np.zeros((160, 160, 3))
        down_and_upsampled_Ref_image = np.zeros((160, 160, 3))
        high_resolution_images_as_references[:height_2, :width_2,
        :] = high_resolution_images_as_references_sub
        down_and_upsampled_Ref_image[:height_2, :width_2,
        :] = down_and_upsampled_Ref_image_sub

        ### change type
        low_resolution_image = low_resolution_image.astype(np.float32)
        upsampled_low_resolution_image = upsampled_low_resolution_image.astype(
            np.float32)
        high_resolution = high_resolution.astype(np.float32)
        high_resolution_images_as_references = high_resolution_images_as_references.astype(
            np.float32)
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.astype(
            np.float32)

        ### rgb range to [-1, 1]
        low_resolution_image = low_resolution_image / 127.5 - 1.
        upsampled_low_resolution_image = upsampled_low_resolution_image / 127.5 - 1.
        high_resolution = high_resolution / 127.5 - 1.
        high_resolution_images_as_references = high_resolution_images_as_references / 127.5 - 1.
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image / 127.5 - 1.

        sample = {
            'low_resolution_image': low_resolution_image,
            'upsampled_low_resolution_image': upsampled_low_resolution_image,
            'high_resolution': high_resolution,
            'high_resolution_images_as_references': high_resolution_images_as_references,
            'down_and_upsampled_Ref_image': down_and_upsampled_Ref_image
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, high_resolution_images_as_references_level='1',
                 transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.png')))
        self.high_resolution_images_as_references_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5',
                                   '*_' + high_resolution_images_as_references_level + '.png')))
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        high_resolution = imread(self.input_list[idx])
        height, width = high_resolution.shape[:2]
        height, width = height // 4 * 4, width // 4 * 4
        high_resolution = high_resolution[:height, :width,
                          :]  ### crop to the multiple of 4

        ### LR and LR_sr
        low_resolution_image = np.array(
            Image.fromarray(high_resolution).resize((width // 4, height // 4),
                                                    Image.BICUBIC))
        upsampled_low_resolution_image = np.array(
            Image.fromarray(low_resolution_image).resize((width, height),
                                                         Image.BICUBIC))

        ### Ref and Ref_sr
        high_resolution_images_as_references = imread(
            self.high_resolution_images_as_references_list[idx])
        height_2, width_2 = high_resolution_images_as_references.shape[:2]
        height_2, width_2 = height_2 // 4 * 4, width_2 // 4 * 4
        high_resolution_images_as_references = high_resolution_images_as_references[
                                               :height_2, :width_2, :]
        down_and_upsampled_Ref_image = np.array(
            Image.fromarray(high_resolution_images_as_references).resize(
                (width_2 // 4, height_2 // 4), Image.BICUBIC))
        down_and_upsampled_Ref_image = np.array(
            Image.fromarray(down_and_upsampled_Ref_image).resize(
                (width_2, height_2), Image.BICUBIC))

        ### change type
        low_resolution_image = low_resolution_image.astype(np.float32)
        upsampled_low_resolution_image = upsampled_low_resolution_image.astype(
            np.float32)
        high_resolution = high_resolution.astype(np.float32)
        high_resolution_images_as_references = high_resolution_images_as_references.astype(
            np.float32)
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.astype(
            np.float32)

        ### rgb range to [-1, 1]
        low_resolution_image = low_resolution_image / 127.5 - 1.
        upsampled_low_resolution_image = upsampled_low_resolution_image / 127.5 - 1.
        high_resolution = high_resolution / 127.5 - 1.
        high_resolution_images_as_references = high_resolution_images_as_references / 127.5 - 1.
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image / 127.5 - 1.

        sample = {
            'low_resolution_image': low_resolution_image,
            'upsampled_low_resolution_image': upsampled_low_resolution_image,
            'high_resolution': high_resolution,
            'high_resolution_images_as_references': high_resolution_images_as_references,
            'down_and_upsampled_Ref_image': down_and_upsampled_Ref_image
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
