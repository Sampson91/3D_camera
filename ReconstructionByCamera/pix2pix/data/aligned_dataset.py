import os

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, options):
        """Initialize this dataset class.

        Parameters:
            options (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, options)
        self.directory_origin_target = os.path.join(options.data_root, options.phase)  # get the image directory
        # get image paths
        self.origin_target_paths = sorted(make_dataset(self.directory_origin_target, options.max_dataset_size))
        # crop_size should be smaller than the size of loaded image
        assert (self.options.load_size >= self.options.crop_size)
        self.input_num_channel = (self.options.output_num_channel if self.options.direction == 'target_to_origin'
                                  else self.options.input_num_channel)
        self.output_num_channel = (self.options.input_num_channel if self.options.direction == 'target_to_origin'
                                   else self.options.output_num_channel)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains origin, target, A_paths and B_paths
            origin (tensor) - - an image in the input domain
            target (tensor) - - its corresponding image in the target domain
            origin_paths (str) - - image paths
            target_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        origin_target_path = self.origin_target_paths[index]
        origin_target = Image.open(origin_target_path).convert('RGB')
        # split origin_target image into origin and target
        widht, height = origin_target.size
        width_half = int(widht / 2)
        origin = origin_target.crop((0, 0, width_half, height))
        target = origin_target.crop((width_half, 0, widht, height))

        # apply the same transform to both origin and target
        transform_params = get_params(self.options, origin.size)
        origin_transform = get_transform(self.options, transform_params, grayscale=(self.input_num_channel == 1))
        target_transform = get_transform(self.options, transform_params, grayscale=(self.output_num_channel == 1))

        origin = origin_transform(origin)
        target = target_transform(target)

        return {'origin': origin, 'target': target,
                'origin_paths': origin_target_path, 'target_paths': origin_target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.origin_target_paths)
