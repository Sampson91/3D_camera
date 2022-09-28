import os
import numpy as np
from PIL import Image
from skimage import transform, color
import pickle
import torch
from torch.utils.data import Dataset
from segm.model.utils import SoftEncodeAB, CIELAB
import argparse

class ImageNet_dataset(Dataset):
    def __init__(self, dataset_directory, img_size=256, split='train'):
        super(ImageNet_dataset, self).__init__()
        self.dataset_directory = dataset_directory
        self.image_directory = os.path.join(self.dataset_directory, split)
        self.image_size = img_size
        self.split = split
        self.filenames = self.load_filenames(self.dataset_directory, split)
        self.number_of_colors = 313

    def load_filenames(self, data_directory, split, filepath='fullfilenames.pickle'):
        if split == 'train':
            split_filepth = os.path.join(data_directory, 'clean_train_filenames.pickle')
        else:
            split_filepth = os.path.join(data_directory, split + '_' + filepath)
        file = open(split_filepth, 'rb')
        filenames = pickle.load(file)
        print('Load from:', split_filepth)
        return filenames

    def resize(self, image, input_size):
        downscale = image.shape[0] > input_size and image.shape[1] > input_size
        resized = transform.resize(image, (input_size, input_size), mode='reflect', anti_aliasing=downscale)   # downscale
        if image.dtype == np.uint8:
            resized *= 255
        else:
            print('input image.dtype is not np.uint8')
            raise NotImplementedError
        return resized.astype(image.dtype)

    def rgb_to_lab(self, image):
        assert image.dtype == np.uint8
        return color.rgb2lab(image).astype(np.float32)

    def numpy_to_torch(self, image):
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0))      # [c, h, w]
        return tensor.type(torch.float32)

    def get_ab(self, key):
        if self.split == 'train':
            image_path = os.path.join(self.image_directory, key[0], key[1])
        else:
            image_path = os.path.join(self.image_directory, key)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_resized = self.resize(image, self.image_size)
        ab_resized = self.rgb_to_lab(image_resized)[:, :, 1:]     # np.float32
        image_ab = self.numpy_to_torch(ab_resized)
        return image_ab


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        key = self.filenames[index]
        image_ab = self.get_ab(key)
        return image_ab, key


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--dataset_directory', type=str, default='/userhome/SUN_text2img/ImageNet')
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_directory = args.dataset_directory
    split = args.split
    default_cielab = CIELAB()
    encode_ab = SoftEncodeAB(default_cielab)
    if split == 'val':
        validation_dataset = ImageNet_dataset(dataset_directory, split='val')
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
        validation_dictionary = {}
        for i, data in enumerate(validation_loader):
            print('i', i)
            image_ab, key = data
            image_ab = image_ab.cuda()
            key = key[0]
            query_actual = encode_ab(image_ab)
            numpy_query_actual = query_actual[0].cpu().detach().numpy()
            validation_dictionary[key] = numpy_query_actual
        validation_file_open = open(os.path.join(dataset_directory, 'val_q_actual.pickle'), 'wb')
        pickle.dump(validation_dictionary, validation_file_open, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        train_dataset = ImageNet_dataset(dataset_directory, split='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        train_dictionary = {}
        for i, data in enumerate(train_loader):
            print(i)
            image_ab, key = data
            image_ab = image_ab.cuda()
            query_actual = encode_ab(image_ab)
            numpy_query_actual = query_actual[0].cpu().detach().numpy()
            train_key = key[0][0] + '_' + key[1][0]
            train_dictionary[train_key] = numpy_query_actual
        train_file_open = open(os.path.join(dataset_directory, 'train_q_actual.pickle'), 'wb')
        pickle.dump(train_dictionary, train_file_open, protocol=pickle.HIGHEST_PROTOCOL)

















