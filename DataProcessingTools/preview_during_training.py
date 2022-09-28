import numpy as np
from PIL import Image
import os
from skimage import io

'''
不同的模型可能需要不同的方法，这里尽可能将所有可以想到的方式都包含到这一个模块里。 
original_image使用list包含以便输入多张np.array形式的图片。
original_image_name输入list。
original_image_path下可以包含子文件夹。
例：ttsr需要lr和ref，这两个都可以放在同一个original_image_path下，
但需要保证lr和对应的ref同名，或者将lr和对应的ref作为list传入original_image_name，[lr, ref]。
'''


def preview(save_path, train_output, epoch, original_image_nparray=None,
            original_image_name=None, original_image_path=None):
    '''
    :param save_path: preview image saving path
    :param train_output: the output from training neural network
    :param epoch: current training epoch for adding to image name
    :param original_image_nparray: input the original image as np.array as a list
    :param original_image_name: input the original image's name as a list
    :param original_image_path: input the directory of the original image
    :return: save directory with file name
    '''

    assert original_image_nparray or original_image_name, 'must get original_image or original_image_name'
    preview = train_output
    final_save_path = None

    if original_image_name:
        assert original_image_path, 'get image name, need image path, as well'
        for name_ in original_image_name:
            for root_, directory_, file_ in os.walk(original_image_path):
                if name_ in file_:
                    original_path_file = os.path.join(root_, name_)
                    real_image = Image.open(original_path_file).convert('RGB')
                    width = train_output.shape[1]
                    height = train_output.shape[0]
                    real_image = real_image.resize((width, height))
                    real_image = np.array(real_image)
                    preview = np.hstack((real_image, preview))
        no_ext_name = []
        ext = None
        for name_ in original_image_name:
            no_ext, ext = os.path.splitext(name_)
            no_ext_name.append(no_ext)
        # no_ext_name.append(ext)
        final_save_path = os.path.join(save_path, str(epoch) + '_' + '_'.join(
            i for i in no_ext_name) + '.jpg')
        io.imsave(final_save_path, preview.astype(np.uint8))

    elif original_image_nparray:
        for real_image_ in original_image_nparray:
            # real_image_ = cv2.resize(real_image_, (train_output[1], train_output[0]))
            real_image_ = Image.fromarray(real_image_)
            width = train_output.shape[1]
            height = train_output.shape[0]
            real_image_ = real_image_.resize((width, height))
            real_image_ = np.array(real_image_)
            preview = np.hstack((real_image_, preview))
        final_save_path = os.path.join(save_path, str(epoch) + '.jpg')
        io.imsave(final_save_path, preview.astype(np.uint8))

    return final_save_path


if __name__ == '__main__':
    save_path = 'C:/Users/Administrator/Desktop/333'
    train_output = (np.ones((160, 160, 3), dtype=np.uint8) * 255)
    original_image_nparray1 = (np.ones((160, 160, 3), dtype=np.uint8) * 100)
    original_image_nparray2 = (np.ones((160, 160, 3), dtype=np.uint8) * 50)
    original_image_nparray = [original_image_nparray1, original_image_nparray2]
    original_image_name = ['gray_0.jpg', 'ref_0.jpg']
    original_image_path = 'C:/Users/Administrator/Desktop/stytr2-1g1c/train-style'
    for epoch in range(2):
        directory = preview(save_path, train_output, epoch,
                            original_image_nparray=original_image_nparray,
                            original_image_name=None,
                            original_image_path=None)
