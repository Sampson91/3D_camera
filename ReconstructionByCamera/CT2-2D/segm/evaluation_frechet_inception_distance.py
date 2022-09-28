import torch
import argparse
from metrics import *
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--prediction_directory', help='colorized images')
    parser.add_argument('--ground_truth_directory', help='groundtruth images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    prediction_directory = args.prediction_directory
    ground_truth_directory = args.ground_truth_directory

    frechet_inception_distance_score, frechet_inception_distance_score_convert = calculate_frechet_inception_distance(
        prediction_directory, ground_truth_directory)
    print('frechet_inception_distance_score', frechet_inception_distance_score,
          'frechet_inception_distance_score_convert:',
          frechet_inception_distance_score_convert)
