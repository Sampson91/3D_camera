import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np


def image_write(path_origin, path_target, path_origin_target):
    image_origin = cv2.imread(path_origin, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    image_origin = cv2.imread(path_target, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    image_origin = np.concatenate([image_origin, image_origin], 1)
    cv2.imwrite(path_origin_target, image_origin)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_origin', dest='fold_origin', help='input directory for image A', type=str, default='')
parser.add_argument('--fold_target', dest='fold_target', help='input directory for image B', type=str, default='')
parser.add_argument('--fold_origin_target', dest='fold_origin_target', help='output directory', type=str, default='')
parser.add_argument('--num_images', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_origin_target', dest='use_origin_target', help='if true: (0001_A, 0001_B) to (0001_AB)',
                    action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing',
                    help='If used, chooses single CPU execution instead of parallel execution',
                    action='store_true', default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_origin)

if not args.no_multiprocessing:
    pool = Pool()

for split in splits:
    image_fold_origin = os.path.join(args.fold_A, split)
    image_fold_target = os.path.join(args.fold_B, split)
    image_list = os.listdir(image_fold_origin)
    if args.use_origin_target:
        image_list = [img_path for img_path in image_list if '_A.' in img_path]

    num_images = min(args.num_imgs, len(image_list))
    print('split = %s, use %d/%d images' % (split, num_images, len(image_list)))
    image_fold_origin_target = os.path.join(args.fold_origin_target, split)
    if not os.path.isdir(image_fold_origin_target):
        os.makedirs(image_fold_origin_target)
    print('split = %s, number of images = %d' % (split, num_images))
    for num in range(num_images):
        name_origin = image_list[num]
        path_origin = os.path.join(image_fold_origin, name_origin)
        if args.use_origin_target:
            name_target = name_origin.replace('_A.', '_B.')
        else:
            name_target = name_origin
        path_target = os.path.join(image_fold_target, name_target)
        if os.path.isfile(path_origin) and os.path.isfile(path_target):
            name_origin_target = name_origin
            if args.use_AB:
                name_origin_target = name_origin_target.replace('_A.', '.')  # remove _A
            path_origin_target = os.path.join(image_fold_origin_target, name_origin_target)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_origin, path_target, path_origin_target))
            else:
                image_origin = cv2.imread(path_origin, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR

                image_target = cv2.imread(path_target, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                image_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
                image_origin_target = np.concatenate([image_origin, image_target], 1)
                cv2.imwrite(path_origin_target, image_origin_target)
if not args.no_multiprocessing:
    pool.close()
    pool.join()
