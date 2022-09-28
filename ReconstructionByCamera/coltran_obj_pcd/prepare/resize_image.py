from PIL import Image
# from PIL.Image import Resampling
import shutil
import os

# copy to another direction
def copy_file(original_path_with_file, destination_path):  # copy file function
    if not os.path.isfile(original_path_with_file):
        print("%s not exist!" % (original_path_with_file))
    else:
        file_path, file_name = os.path.split(original_path_with_file)  # split file name and path
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)  # 创建路径
        shutil.copy(original_path_with_file, destination_path + '/' + file_name)  # 复制文件
        # print("copy %s -> %s" % (original_path, destination_path + file_name))


def resize_image_function(image_input, image_output, pixel):
    """
    改变图片大小
    :param image_input: 输入图片
    :param image_output: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """
    opened_image = Image.open(image_input).convert('RGB')
    width = int(pixel)
    height = int(pixel)
    type = opened_image.format
    out = opened_image.resize((width, height))
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out.save(image_output, type)


################################################################################
#####                        convert argb to rgb                          ######
################################################################################
def argb_convert_to_rgb(path_with_file):
    opened_image = Image.open(path_with_file).convert('RGB')
    type = opened_image.format
    opened_image.save(path_with_file, type)


if __name__ == '__main__':

    scale = 512 // 64
    original_path = 'C:/Users/Administrator/Desktop/test/BFM2017.jpeg'
    file_name='BFM2017.png'
    # destination_path = './upscaled_image'
    #
    #
    # original_path_with_file = os.path.join(original_path, file_name)
    # resize_path = os.path.join(destination_path, file_name)
    # copy_file(original_path_with_file, destination_path)
    # resize_image_function(resize_path, resize_path, pixel=scale)

    argb_convert_to_rgb(path_with_file = original_path)
