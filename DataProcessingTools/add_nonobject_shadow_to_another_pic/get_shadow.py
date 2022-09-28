import os

import PIL
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage.morphology import remove_small_objects


# use dilation and erosion to get shadow, not good
def deal_shadow(input_path_file, output_path_file):
    image = cv2.imread(input_path_file, 0)
    kernel = np.ones((3, 3), np.uint8)

    retvalve, threshold1 = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY_INV)

    # 闭运算
    dilation = cv2.dilate(threshold1, kernel, iterations=3)  # 膨胀
    erosion = cv2.erode(dilation, kernel, iterations=1)  # 腐蚀
    minus_erosion = -erosion + 255

    cv2.imwrite(output_path_file, minus_erosion)


# not good
def deal_shadow_2(input_path_file, output_path_file):
    image = cv2.imread(input_path_file)
    image_sum = np.sum(image, axis=2)
    standard_image = np.std(image)

    standard_image = np.where(image_sum < 150, standard_image, 255)
    # cv2.imwrite(output_path_file, std_img.astype(np.uint8))
    # plt.imshow(std_img)
    cv2.imshow('5', standard_image)
    cv2.waitKey(0)


# by keeping dark gray pixels
def deal_shadow_3(input_path_file, output_path_file):
    image = cv2.imread(input_path_file, 0)
    original_image = copy.deepcopy(image)
    # 计算灰白色部分像素的均值（越小非shadow部分颜色越深）
    # pixel = int(np.mean(img[img > 200]))
    # 把灰白色部分修改为与背景接近的颜色(越大保留的细节越多)
    # img[img < 35] = pixel
    # 255 非shadow部分就纯白，用pixel就根据上边算的不一定纯白
    image[image > 60] = 255  # pixel

    cv2.imwrite(output_path_file, image)


def deal_shadow_4(input_path_file, output_path_file):  # 非常差
    image = input_path_file
    image = cv2.imread(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换为HSV三通道图像
    H, S, V = cv2.split(image_hsv)  # 将三通道分开
    # m = np.double(((S - V) / (H + S + V)))
    # img_nos = np.where(m, 255, img_gray)

    cv2.imwrite(output_path_file, image_hsv)


def directory_exists_or_create(directory):
    if not os.path.isdir(directory):
        total_directory = os.path.join(os.getcwd(), directory)
        os.mkdir(total_directory)


def main_obtain_shadow_keep_name(input_path, output_path):
    image_list = os.listdir(input_path)
    assert os.path.isdir(input_path), 'input image directory is not exist'
    assert len(image_list), 'no image in input directory.'
    directory_exists_or_create(output_path)
    for image_ in image_list:
        image_path_file = os.path.join(input_path, image_)
        output_path_file = os.path.join(output_path, image_)
        deal_shadow_3(image_path_file, output_path_file)


def main_combine_sub_folder_name(input_path, output_path):
    assert os.path.isdir(input_path), 'input image directory is not exist'
    directory_exists_or_create(output_path)
    for root, directory, list_file in os.walk(input_path):
        # list_file = os.listdir(file_direction)  # 返回指定目录
        i = 0
        for file in list_file:
            file_name, ext = os.path.splitext(file)  # 返回文件名和后缀
            last_folder = os.path.basename(root)
            newfile = last_folder + '_' + str(i) + ext
            i += 1
            input_path_file = os.path.join(root, file)
            output_path_file = os.path.join(output_path, newfile)
            deal_shadow_3(input_path_file, output_path_file)


if __name__ == '__main__':
    input_path = 'C:/Users/Administrator/Desktop/11/cut'  # kt/ori'
    output_path = 'C:/Users/Administrator/Desktop/11/shadow'  # kt/did'
    main_obtain_shadow_keep_name(input_path, output_path)
    print('shadows are obtained from non-objects images')
    # input_path_file = 'C:/Users/Administrator/Desktop/cgtrader/ref/ref_0.jpg'
    # output_path_file = 'C:/Users/Administrator/Desktop/xxx.jpg'
    # deal_shadow_4(input_path_file, output_path_file)
