import argparse
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

NUM_FILES_PER_SAMPLE = 6
NUM_SEGMENTS = 8


def args_parser():
    parser = argparse.ArgumentParser(
        description="Process input images and creates training and cross validation data sets")
    parser.add_argument("--in-dir", type=str, required=True, help="Input directory of images for classification")
    parser.add_argument("--target-dir", type=str, required=True, help="Output directory for target data set")
    parser.add_argument("--train-dir", type=str, required=True, help="Training directory data set")
    parser.add_argument("--xv-dir", type=str, required=False, help="Cross-validation directory data set")
    parser.add_argument("--xv-sample-rate", type=float, required=False, choices=np.arange(0.1, 0.4, 0.05),
                        help="Cross-Validation rate to be taken from training set")
    return parser.parse_args()


def split_file_name(file_nm):
    file_nm_parts = re.split('[_.]', file_nm)
    site_id = file_nm_parts[0]
    tile_number = file_nm_parts[2]
    file_type = file_nm_parts[3]
    file_extension = file_nm_parts[4]
    return site_id, tile_number, file_type, file_extension


def split_image_segments(src_image, num_segments, saving_path):
    if src_image.shape[0] % num_segments != 0:
        RuntimeError("The number of tiles provided is not suitable to divide image height '(%i mod %i is not zero)'" %
                     src_image.shape[0] % num_segments)
    segment_height = src_image.shape[0] / num_segments

    if src_image.shape[0] % num_segments != 0:
        RuntimeError("The number of tiles provided is not suitable to divide image width '(%i mod %i is not zero)'" %
                     src_image.shape[1] % num_segments)
    segment_width = src_image.shape[1] / num_segments

    processing_segment = 1
    for i in range(num_segments):
        if len(src_image.shape) == 3:
            this_image = src_image[int(i * segment_height):int(processing_segment * segment_height),
                         int(i * segment_width):int(processing_segment * segment_width), :]
        else:
            this_image = src_image[int(i * segment_height):int(processing_segment * segment_height),
                         int(i * segment_width):int(processing_segment * segment_width)]
        np.save("{}_{}".format(saving_path, str(processing_segment).zfill(3)), this_image)
        processing_segment += 1
    return


def create_binary_gt(input_matrix):
    binary_gt = np.ones(input_matrix.shape, dtype=np.uint8)
    binary_gt[input_matrix == 2] = 0
    return binary_gt


arguments = args_parser()
if not os.path.isdir(arguments.in_dir):
    raise RuntimeError("Input directory does not exist '%s'" % arguments.in_dir)
if not os.path.isdir(arguments.train_dir):
    os.makedirs(arguments.train_dir)
if not os.path.isdir(arguments.target_dir):
    os.makedirs(arguments.target_dir)
if arguments.create_xv:
    if not os.path.isdir(arguments.xv_dir):
        os.makedirs(arguments.xv_dir)

files = os.listdir(arguments.in_dir)
num_samples = len(files) / NUM_FILES_PER_SAMPLE
if arguments.create_xv:
    training_factor = 1 - arguments.xv_sample_rate
    num_training_samples = training_factor * num_samples
    num_xv_samples = arguments.xv_sample_rate * num_samples
else:
    num_training_samples = num_samples

sites_tiles = {}
sample_number = 0
for file_name in files:
    if file_name == '.DS_Store':
        continue

    site, tile, f_type, f_ext = split_file_name(file_name)

    current_key = site + "_" + tile
    if current_key not in sites_tiles.keys():
        sites_tiles[current_key] = {}

    if f_type in ["DSM", "DTM", "RGB", "GTL"]:
        if f_type == "DSM":
            processed_image = cv2.imread(os.path.join(arguments.in_dir, file_name), cv2.IMREAD_UNCHANGED)
        elif f_type == "DTM":
            processed_image = cv2.imread(os.path.join(arguments.in_dir, file_name), cv2.IMREAD_UNCHANGED)
        elif f_type == "RGB":
            processed_image = cv2.imread(os.path.join(arguments.in_dir, file_name), cv2.IMREAD_COLOR)
        elif f_type == "GTL":
            processed_image = cv2.imread(os.path.join(arguments.in_dir, file_name), cv2.IMREAD_GRAYSCALE)
        if sites_tiles[current_key] is not None:
            sites_tiles[current_key][f_type] = processed_image
            stored_images = sites_tiles[current_key]
            if len(stored_images.keys()) == 4:
                dsm_img = stored_images["DSM"]
                dtm_img = stored_images["DTM"]
                rgb_img = stored_images["RGB"]
                gt_image = stored_images["GTL"]
                height_matrix = dsm_img - dtm_img
                final_image = np.dstack((rgb_img, height_matrix))

                saving_dir = arguments.train_dir
                if sample_number <= num_training_samples:
                    saving_dir = arguments.train_dir
                else:
                    saving_dir = arguments.xv_dir

                sample_number += 1
                sites_tiles[current_key] = None
                split_image_segments(final_image, NUM_SEGMENTS,
                                     os.path.join(saving_dir, "{}_{}".format(site, tile)))
                y = create_binary_gt(gt_image)
                split_image_segments(y, NUM_SEGMENTS,
                                     os.path.join(arguments.target_dir, "{}_{}".format(site, tile)))
