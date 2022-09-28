import os
import numpy as np
import cv2
from PIL import Image


# add two images by weight
def add_shadow(input_image_shadow, input_image_rgb, output_image=None):
    shadow = cv2.imread(input_image_shadow)
    image = cv2.imread(input_image_rgb)
    # shadow = np.array(shadow)
    # image = np.array(image)
    # added_shadow = np.clip(image - (255- shadow), 0, 255)
    added_shadow = cv2.addWeighted(image, 0.9, shadow, 0.1, gamma=5)
    # cv2.imshow('1', added_shadow)
    # cv2.waitKey(0)
    cv2.imwrite(output_image, added_shadow)


# add non-white pixels by weight, better
def add_shadow_2(input_image_shadow, input_image_rgb, ratio_of_origin_image,
                 output_image=None):
    shadow = cv2.imread(input_image_shadow, 0)
    image = cv2.imread(input_image_rgb)
    added_shadow = image.copy()
    # shadow = np.array(shadow)
    # image = np.array(image)
    # added_shadow = np.clip(image - (255- shadow), 0, 255)
    for i in range(shadow.shape[0]):
        for j in range(shadow.shape[1]):
            if shadow[i, j] != 255:
                added_shadow[i, j] = np.clip(
                    (ratio_of_origin_image * image[i, j] - (
                            1 - ratio_of_origin_image) * shadow[i, j]), 0, 255)
    # cv2.imshow('1', added_shadow)
    # cv2.waitKey(0)
    cv2.imwrite(output_image, added_shadow)


def directory_exists_or_create(directory):
    if not os.path.isdir(directory):
        total_directory = os.path.join(os.getcwd(), directory)
        os.mkdir(total_directory)


def main_add_shadow(input_path_shadow, input_path_rgb, output_path):
    shadow_list = os.listdir(input_path_shadow)
    assert os.path.isdir(
        input_path_shadow), 'input shadow image directory is not exist'
    assert len(shadow_list), 'no shadow image in input directory.'
    rgb_list = os.listdir(input_path_rgb)
    assert os.path.isdir(
        input_path_rgb), 'input rgb image directory is not exist'
    assert len(rgb_list), 'no rgb image in input directory.'

    directory_exists_or_create(output_path)

    for shadow_ in shadow_list:
        for rgb_ in rgb_list:
            if shadow_ == rgb_:
                input_image_shadow = os.path.join(input_path_shadow, shadow_)
                input_image_rgb = os.path.join(input_path_rgb, rgb_)
                output_image = os.path.join(output_path, rgb_)
                add_shadow_2(input_image_shadow, input_image_rgb,
                             ratio_of_origin_image=0.7,
                             output_image=output_image)


if __name__ == '__main__':
    input_path_shadow = 'C:/Users/Administrator/Desktop/11/shadow'
    input_path_rgb = 'C:/Users/Administrator/Desktop/11/blank_model'
    output_path = 'C:/Users/Administrator/Desktop/11/added_shadow'
    main_add_shadow(input_path_shadow, input_path_rgb, output_path)
    print('all shadows are added to images')
