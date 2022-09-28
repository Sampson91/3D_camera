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

    learned_perceptual_image_patch_similarity, learned_perceptual_image_patch_similarity_convert = average_learned_perceptual_image_patch_similarity(
        prediction_directory, ground_truth_directory)
    structure_similarity_index_measure, peak_signal_to_noise_ratio, structure_similarity_index_measure_convert, peak_signal_to_noise_ratio_convert = average_structure_similarity_index_measure_peak_signal_to_noise_ratio(
        prediction_directory, ground_truth_directory)

    print('structure_similarity_index_measure:',
          structure_similarity_index_measure,
          "structure_similarity_index_measure_convert:",
          structure_similarity_index_measure_convert,
          'peak_signal_to_noise_ratio:',
          peak_signal_to_noise_ratio, 'peak_signal_to_noise_ratio_convert:',
          peak_signal_to_noise_ratio_convert,
          'learned_perceptual_image_patch_similarity:',
          learned_perceptual_image_patch_similarity,
          'learned_perceptual_image_patch_similarity_convert:',
          learned_perceptual_image_patch_similarity_convert)
