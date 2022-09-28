from PIL import Image
import numpy as np
import os
import time


def sketch_image(input_path, output_path, depths=25):
    gray_image = np.asarray(Image.open(input_path).convert('L')).astype('float')
    depth = depths  # (0-100)
    gradient = np.gradient(gray_image)  # 取图像灰度的梯度值
    gradient_x, gradient_y = gradient  # 分别取横纵图像梯度值
    gradient_x = gradient_x * depth / 100.
    gradient_y = gradient_y * depth / 100.
    square_gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1.)
    uniform_x = gradient_x / square_gradient
    uniform_y = gradient_y / square_gradient
    uniform_z = 1. / square_gradient
    angle_radian = np.pi / 2.2  # 光源的俯视角度，弧度值
    azimuth_radian = np.pi / 4.  # 光源的方位角度，弧度值
    influence_x = np.cos(angle_radian) * np.cos(azimuth_radian)  # 光源对x 轴的影响
    influence_y = np.cos(angle_radian) * np.sin(azimuth_radian)  # 光源对y 轴的影响
    influence_z = np.sin(angle_radian)  # 光源对z 轴的影响
    normalization = 255 * (influence_x * uniform_x + influence_y * uniform_y + influence_z * uniform_z)  # 光源归一化
    normalization = normalization.clip(0, 255)
    image = Image.fromarray(normalization.astype('uint8'))  # 重构图像
    image.save(output_path)


def main(input_path, output_path, depths):
    # check output path, if not exist, create
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('create output directory', output_path)
    # read all images from input_path
    images = os.listdir(input_path)
    for image_ in images:
        input_path_with_name = os.path.join(input_path, image_)
        output_path_with_name = os.path.join(output_path, image_)
        sketch_image(input_path=input_path_with_name,
                     output_path=output_path_with_name,
                     depths=depths)
        # 25 is the best
        print('image output ->', output_path_with_name)


if __name__ == '__main__':
    input_path = 'C:/Users/Administrator/Desktop/rgb_image_folder'
    output_path = 'C:/Users/Administrator/Desktop/sketch_image_folder'
    main(input_path, output_path, depths=25)  # depth 越大信息越多
