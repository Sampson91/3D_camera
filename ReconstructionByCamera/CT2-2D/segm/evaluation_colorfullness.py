from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--prediction_directory', help='colorized images')
    args = parser.parse_args()
    return args

def image_colorfulness(image):
    (blue, green, red) = cv2.split(image.astype('float'))
    red_green = np.absolute(red - green)
    yellow_blue = np.absolute(0.5 * (red+green) - blue)
    (red_blue_mean, red_blue_standard_deviation) = (np.mean(red_green), np.std(red_green))
    (yellow_blue_mean, yellow_blue_standard_deviation) = (np.mean(yellow_blue), np.std(yellow_blue))
    standard_deviation_root = np.sqrt((red_blue_standard_deviation ** 2) + (yellow_blue_standard_deviation ** 2))
    mean_root = np.sqrt((red_blue_mean ** 2) + (yellow_blue_mean ** 2))
    return standard_deviation_root + (0.3 * mean_root)

if __name__ == '__main__':
    args = parse_args()
    prediction_directory = args.prediction_directory
    filename_list = os.listdir(prediction_directory)
    assert len(filename_list) == 5000
    total_colorfullness = 0
    count = 0
    for image_path_ in filename_list:
        print('count', count)
        image_path = os.path.join(prediction_directory, image_path_)
        image = cv2.imread(image_path)
        color = image_colorfulness(image)
        total_colorfullness += color
        count += 1
    colorfullness = total_colorfullness/len(filename_list)

