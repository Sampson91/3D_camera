import os
import cv2
import numpy as np


def min_filter_gray(source, radius=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(source, np.ones((2 * radius + 1, 2 * radius + 1)))


def guided_filter(input, patch, radius, eps):
    height, width = input.shape
    image_input = cv2.boxFilter(input, -1, (radius, radius))
    image_patch = cv2.boxFilter(patch, -1, (radius, radius))
    image_input_patch = cv2.boxFilter(input * patch, -1, (radius, radius))
    calculated_input_patch = image_input_patch - image_input * image_patch

    image_input_square = cv2.boxFilter(input * input, -1, (radius, radius))
    difference_square = image_input_square - image_input * image_input

    divided = calculated_input_patch / (difference_square + eps)
    difference = image_patch - divided * image_input

    image_divided_box = cv2.boxFilter(divided, -1, (radius, radius))
    image_difference_box = cv2.boxFilter(difference, -1, (radius, radius))
    return image_divided_box * input + image_difference_box


def defog(image, radius, eps, defog_level, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    dark_channal = np.min(image, 2)  # 得到暗通道图像
    Dark_Channel = min_filter_gray(dark_channal, 1)
    # cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    dark_channal = guided_filter(dark_channal, Dark_Channel, radius,
                                 eps)  # 使用引导滤波优化
    bins = 2000
    light = np.histogram(dark_channal, bins)  # 计算大气光照A
    divide = np.cumsum(light[0]) / float(dark_channal.size)
    for lmax in range(bins - 1, 0, -1):
        if divide[lmax] <= 0.999:
            break
    lighter = np.mean(image, 2)[dark_channal >= light[1][lmax]].max()
    dark_channal = np.minimum(dark_channal * defog_level, maxV1)  # 对值范围进行限制
    return dark_channal, lighter


def dehaze(image, radius=81, eps=0.001, defog_level=0.95, maxV1=0.80,
           bGamma=True, gamma=0.5):
    # defog_level 去雾的程度
    # 暗通道最小值滤波半径radius
    zeros_image_to_correct_color = np.zeros(image.shape)
    mask_image, light = defog(image, radius, eps, defog_level,
                              maxV1)  # 得到遮罩图像和大气光照

    for i in range(3):
        zeros_image_to_correct_color[:, :, i] = (image[:, :,
                                                 i] - mask_image) / (
                                                            1 - mask_image / light)  # 颜色校正
    zeros_image_to_correct_color = np.clip(zeros_image_to_correct_color, 0, 1)
    if bGamma:
        zeros_image_to_correct_color = zeros_image_to_correct_color ** (
                    np.log(gamma) / np.log(
                zeros_image_to_correct_color.mean()))  # gamma校正,默认不进行该操作
    return zeros_image_to_correct_color


def main(input_path, output_path):
    image_list = os.listdir(input_path)
    for image_ in image_list:
        image_path_file = os.path.join(input_path, image_)
        # 0 < gamma < 1, defog_level 去雾的程度, radius 暗通道最小值滤波半径
        image = dehaze(cv2.imread(image_path_file) / 255.0, radius=100,
                       defog_level=0.9,
                       bGamma=True, gamma=0.46) * 255
        image_output = os.path.join(output_path, image_)
        print(image_output)
        cv2.imwrite(image_output, image)


if __name__ == '__main__':
    input_path = 'C:/Users/Administrator/Desktop/cgtrader/test'
    output_path = 'C:/Users/Administrator/Desktop/cgtrader/did'
    main(input_path, output_path)
