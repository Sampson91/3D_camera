import os

from PIL import Image


def rgb_to_gray_function(path, file, out_path):
    total_path = os.path.join(path, file)

    rgb_image = Image.open(total_path).convert('RGB')
    gray_image = rgb_image.convert('L')

    gray_saving_path = os.path.join(out_path, file)
    gray_image.save(gray_saving_path)
