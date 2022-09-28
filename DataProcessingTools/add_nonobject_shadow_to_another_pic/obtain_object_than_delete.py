import os

import pixellib
from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation
import cv2
import numpy as np


# 获取带蒙版的图片
def obatin_objects(input_path_file):  # , output_path_file):
    segment_image = instance_segmentation()  # semantic_segmentation()
    segment_image.load_model("h5/mask_rcnn_coco.h5")  # _pascalvoc
    # cv2.imshow('segment_image', segment_image)
    # cv2.waitKey(0)
    r, output = segment_image.segmentImage(
        input_path_file)  # ,output_image_name=output_path_file)  # , mask_points_values=True, save_extracted_objects=True)  # segmentAsPascalvoc
    # output == saved image
    # cv2.imshow('output', output)
    # cv2.waitKey(0)
    return output


# 另一种方式直接获取蒙版，但效果不好，弃用
def obatin_objects_2(input_path_file, output_path_file):
    segment_image = semantic_segmentation()
    segment_image.load_pascalvoc_model(
        "h5/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    segment_image.segmentAsPascalvoc(input_path_file,
                                     output_image_name=output_path_file)
    # 带有分段叠加层的图像，只需要将segmentAsPascalvoc()函数的overlay属性设置为True


# 原图减去带蒙版的图片，获取蒙版
def image_defference(input_original_image, input_semantic_image,
                     output_path_image):
    original_image = cv2.imread(input_original_image)
    # semantic_image = cv2.imread(input_semantic_image)
    semantic_image = input_semantic_image
    differences = abs(original_image - semantic_image)

    # cv2.imshow('differences', differences)
    # cv2.waitKey(0)
    cv2.imwrite(output_path_image, differences)
    return differences


# 从原图减去目标
def delete_objects(input_original_image, input_semantic_image,
                   output_path_file):
    original_image = cv2.imread(input_original_image)
    semantic_image = cv2.imread(input_semantic_image, 0)
    uv = []
    for u_ in range(semantic_image.shape[0]):
        for v_ in range(semantic_image.shape[1]):
            if semantic_image[u_, v_] != 0:
                uv.append([u_, v_])
    for uv_ in uv:
        original_image[uv_[0], uv_[1]] = 255
    # cv2.imshow('original_image', original_image)
    # cv2.waitKey(0)
    cv2.imwrite(output_path_file, original_image)


def directory_exists_or_create(directory):
    if not os.path.isdir(directory):
        total_directory = os.path.join(os.getcwd(), directory)
        os.mkdir(total_directory)


def main_delete_objects(input_path, output_path, delete_output_path,
                        object_only_path):
    image_list = os.listdir(input_path)
    assert os.path.isdir(input_path), 'input image directory is not exist'
    assert len(image_list), 'no image in input directory.'
    for image_ in image_list:
        input_path_file = os.path.join(input_path, image_)
        output_path_file = os.path.join(output_path, image_)

        # check and create directories
        directory_exists_or_create(output_path)
        directory_exists_or_create(delete_output_path)
        directory_exists_or_create(object_only_path)

        semantic_image = obatin_objects(input_path_file)  # , output_path_file)
        delete_output_path_file = os.path.join(delete_output_path, image_)
        object_only_output_path_image = os.path.join(object_only_path, image_)
        image_defference(input_original_image=input_path_file,
                         input_semantic_image=semantic_image,
                         output_path_image=object_only_output_path_image)
        delete_objects(input_original_image=input_path_file,
                       input_semantic_image=object_only_output_path_image,
                       output_path_file=delete_output_path_file)


if __name__ == '__main__':
    input_path = 'C:/Users/Administrator/Desktop/1/kt'
    output_path = 'C:/Users/Administrator/Desktop/1/semantic'
    delete_output_path = 'C:/Users/Administrator/Desktop/1/cut'
    object_only_path = 'C:/Users/Administrator/Desktop/1/object_only'
    main_delete_objects(input_path, output_path, delete_output_path,
                        object_only_path)
    print('all objects are deleted from images')
