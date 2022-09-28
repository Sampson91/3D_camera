"""A modified image folder class

We modify the official PyTorch image folder
(https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os

import torch.utils.data as data
from PIL import Image

IMAGE_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)


def make_dataset(directory, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, file_names in sorted(os.walk(directory)):
        for file_name in file_names:
            if is_image_file(file_name):
                path = os.path.join(root, file_name)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        images = make_dataset(root)
        if len(images) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "Supported image extensions are: " + ",".join(
                IMAGE_EXTENSIONS)))

        self.root = root
        self.images = images
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.return_paths:
            return image, path
        else:
            return image

    def __len__(self):
        return len(self.images)
