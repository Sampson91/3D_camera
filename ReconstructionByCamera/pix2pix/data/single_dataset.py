from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, options):
        """Initialize this dataset class.

        Parameters:
            options (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, options)
        self.origin_paths = sorted(make_dataset(options.dataroot, options.max_dataset_size))
        input_num_channel = (self.options.output_num_channel if self.options.direction == 'target_to_origin'
                             else self.options.input_num_channel)
        self.transform = get_transform(options, grayscale=(input_num_channel == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains origin and A_paths
            origin(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        origin_path = self.origin_paths[index]
        origin_image = Image.open(origin_path).convert('RGB')
        origin = self.transform(origin_image)
        return {'origin': origin, 'A_paths': origin_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.origin_paths)
