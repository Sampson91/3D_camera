import numbers
import os
import numpy as np
import glob
import random

import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional
from torch.utils.data import Dataset
from torchvision import transforms
from cloudreader import ReadObj
from cloudreader import read_pcd

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def random_point_dropout(pointcloud, dropout_ratio=0.75):
    "Shape: N, C"
    size = int((pointcloud.shape[0]) * dropout_ratio)
    drop_idx = np.random.choice(pointcloud.shape[0], size, replace=False)
    pointcloud = np.delete(pointcloud, drop_idx, 0) 
    return pointcloud


def cloud_upsample(pointcloud):
    "Shape: N, C"
    pointcloud = pointcloud.transpose((1, 0)) #N, C->C, N
    pointcloud = torch.unsqueeze(torch.from_numpy(pointcloud), 0)
    pointcloud = torch_neural_network_functional.interpolate(pointcloud, size=None, scale_factor=4, mode='linear', align_corners=None)
    pointcloud = torch.squeeze(pointcloud, 0)
    pointcloud = pointcloud.numpy()
    pointcloud = pointcloud.transpose((1, 0)) # C, N->N, C
    return pointcloud


class ToTensor(object):
    def __call__(self, sample):
        low_resolution_image, upsampled_low_resolution_image, high_resolution, high_resolution_images_as_references, down_and_upsampled_Ref_image = \
            sample['low_resolution_image'], sample[
                'upsampled_low_resolution_image'], sample['high_resolution'], \
            sample['high_resolution_images_as_references'], sample[
                'down_and_upsampled_Ref_image']
        # For image H, W, C->C, H, W; For Cloud, from N, C->C, N
        low_resolution_image = low_resolution_image.transpose((1, 0))
        upsampled_low_resolution_image = upsampled_low_resolution_image.transpose(
            (1, 0))
        high_resolution = high_resolution.transpose((1, 0))
        high_resolution_images_as_references = high_resolution_images_as_references.transpose(
            (1, 0))
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.transpose(
            (1, 0))
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
    def __init__(self, args, transform=transforms.Compose([ToTensor()])): # Delete RandomFlip() and RandomRotate()
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
        # HR
        self.obj = ReadObj(self) # To read obj files
        if self.input_list[idx].endswith(".obj"):
            high_resolution = self.obj.cloud_read(self.input_list[idx])
        else:
            high_resolution = read_pcd(self.input_list[idx])
        numbers = high_resolution.shape[0]
        # numbers = numbers//4*4 # We don't know why this operation
        
        # LR and LR_sr
        low_resolution_image = random_point_dropout(high_resolution)
        upsampled_low_resolution_image = cloud_upsample(low_resolution_image)

        # Ref and Ref_sr
        if self.input_list[idx].endswith(".obj"):
            high_resolution_images_as_references_sub = self.obj.cloud_read(
                self.high_resolution_images_as_references_list[idx])
        else:
            high_resolution_images_as_references_sub = read_pcd(self.high_resolution_images_as_references_list[idx])
        numbers_2 = high_resolution_images_as_references_sub.shape[0]
        down_and_upsampled_Ref_image_sub = random_point_dropout(high_resolution_images_as_references_sub)
        down_and_upsampled_Ref_image_sub = cloud_upsample(down_and_upsampled_Ref_image_sub)

        # complete ref and ref_sr to the same size, to use batch_size > 1
        high_resolution_images_as_references = np.zeros((numbers, 6)) # Need to check/update the number of cloud point N
        down_and_upsampled_Ref_image = np.zeros((numbers, 6))
        high_resolution_images_as_references[:numbers_2, :] = high_resolution_images_as_references_sub
        # down_and_upsampled_Ref_image[:numbers_2, :] = down_and_upsampled_Ref_image_sub # this line should be used in release version
        down_and_upsampled_Ref_image[:numbers_2, :] = down_and_upsampled_Ref_image_sub # this line should not be used in release version

        # change type
        low_resolution_image = low_resolution_image.astype(np.float32)
        upsampled_low_resolution_image = upsampled_low_resolution_image.astype(
            np.float32)
        high_resolution = high_resolution.astype(np.float32)
        high_resolution_images_as_references = high_resolution_images_as_references.astype(
            np.float32)
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.astype(
            np.float32)

        # rgb range to [-1, 1]
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
    """
    This class is for eval model, now it is can not take effect, need to recode, 
    """
    def __init__(self, args):
        self.input_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, 'test/lr', '*.obj')))
        self.high_resolution_images_as_references_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, 'test/ref','*.obj')))


    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # HR
        self.obj = ReadObj(self.input_list[idx])

        # LR and LR_sr
        low_resolution_image = self.obj.cloud_read(self.input_list[idx])
        number_1 = low_resolution_image.shape[0]
        upsampled_low_resolution_image = cloud_upsample(low_resolution_image)

        # Ref and Ref_sr
        high_resolution_images_as_references = self.obj.cloud_read(self.high_resolution_images_as_references_list[idx])
        number_2 = high_resolution_images_as_references.shape[0]
        number_2 = number_2 // 4 * 4
        high_resolution_images_as_references = high_resolution_images_as_references[:number_2 :]
        down_and_upsampled_Ref_image = random_point_dropout(high_resolution_images_as_references)
        down_and_upsampled_Ref_image = cloud_upsample(down_and_upsampled_Ref_image)

        # change type
        low_resolution_image = low_resolution_image.astype(np.float32)
        upsampled_low_resolution_image = upsampled_low_resolution_image.astype(
            np.float32)
        high_resolution_images_as_references = high_resolution_images_as_references.astype(
            np.float32)
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.astype(
            np.float32)

        # rgb range to [-1, 1]
        low_resolution_image = low_resolution_image / 127.5 - 1.
        upsampled_low_resolution_image = upsampled_low_resolution_image / 127.5 - 1.
        high_resolution_images_as_references = high_resolution_images_as_references / 127.5 - 1.
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image / 127.5 - 1.

        # to tensor
        low_resolution_image = torch.from_numpy(
            low_resolution_image.transpose((1, 0))).float()
        upsampled_low_resolution_image = torch.from_numpy(
            upsampled_low_resolution_image.transpose((1, 0))).float()
        high_resolution_images_as_references = torch.from_numpy(
            high_resolution_images_as_references.transpose(
                (1, 0))).float()
        down_and_upsampled_Ref_image = torch.from_numpy(
            down_and_upsampled_Ref_image.transpose((1, 0))).float()

        sample = {
            'low_resolution_image': low_resolution_image,
            'upsampled_low_resolution_image': upsampled_low_resolution_image,
            'high_resolution_images_as_references': high_resolution_images_as_references,
            'down_and_upsampled_Ref_image': down_and_upsampled_Ref_image
        }

        return sample


class InferenceSet(Dataset):
    def __init__(self, args):
        self.input_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, 'test/lr', '*.obj')))
        self.high_resolution_images_as_references_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, 'test/ref','*.obj')))   

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # HR
        self.obj = ReadObj(self.input_list[idx])
        
        ### LR and LR_sr
        low_resolution_image = self.obj.cloud_read(self.input_list[idx])
        number_1 = low_resolution_image.shape[0]
        upsampled_low_resolution_image = cloud_upsample(low_resolution_image)

        ### Ref and Ref_sr
        high_resolution_images_as_references = self.obj.cloud_read(self.high_resolution_images_as_references_list[idx])
        number_2 = high_resolution_images_as_references.shape[0]
        number_2 = number_2 // 4 * 4
        high_resolution_images_as_references = high_resolution_images_as_references[:number_2 :]
        down_and_upsampled_Ref_image = random_point_dropout(high_resolution_images_as_references)
        down_and_upsampled_Ref_image = cloud_upsample(down_and_upsampled_Ref_image)

        ### change type
        low_resolution_image = low_resolution_image.astype(np.float32)
        upsampled_low_resolution_image = upsampled_low_resolution_image.astype(
            np.float32)
        high_resolution_images_as_references = high_resolution_images_as_references.astype(
            np.float32)
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.astype(
            np.float32)

        ### rgb range to [-1, 1]
        low_resolution_image = low_resolution_image / 127.5 - 1.
        upsampled_low_resolution_image = upsampled_low_resolution_image / 127.5 - 1.
        high_resolution_images_as_references = high_resolution_images_as_references / 127.5 - 1.
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image / 127.5 - 1.

        ### to tensor
        low_resolution_image = torch.from_numpy(
            low_resolution_image.transpose((1, 0))).float()
        upsampled_low_resolution_image = torch.from_numpy(
            upsampled_low_resolution_image.transpose((1, 0))).float()
        high_resolution_images_as_references = torch.from_numpy(
            high_resolution_images_as_references.transpose(
                (1, 0))).float()
        down_and_upsampled_Ref_image = torch.from_numpy(
            down_and_upsampled_Ref_image.transpose((1, 0))).float()

        sample = {
            'low_resolution_image': low_resolution_image,
            'upsampled_low_resolution_image': upsampled_low_resolution_image,
            'high_resolution_images_as_references': high_resolution_images_as_references,
            'down_and_upsampled_Ref_image': down_and_upsampled_Ref_image
        }

        return sample