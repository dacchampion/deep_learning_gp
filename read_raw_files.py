import argparse
import os
import re
import numpy as np
import cv2

NUM_FILES_PER_SAMPLE = 6
EXPECTED_DIM = 2048
CROP_DIM = 256
OVERLAP = 0.5


def args_parser():
    parser = argparse.ArgumentParser(
        description="Process input images and creates training and cross validation data sets")
    parser.add_argument("--in-dir", type=str, required=True, help="Input directory of images for classification")
    parser.add_argument("--out-dir", type=str, required=False, help="Output directory just for input processing")
    parser.add_argument("--target-dir", type=str, required=False, help="Directory for target data set")
    parser.add_argument("--train-dir", type=str, required=False, help="Directory for training data set")
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


def crop_image(src_image, saving_path):
    pieces = []
    start_pts = np.linspace(0, EXPECTED_DIM, EXPECTED_DIM / (OVERLAP * CROP_DIM) + 1)
    start_pts = start_pts[:-2]
    start_pts = [int(x) for x in start_pts]
    for i in start_pts:
        for j in start_pts:
            if src_image.shape==3:
                croped_image = src_image[i:i + CROP_DIM, j:j + CROP_DIM, :]
            else:
                croped_image = src_image[i:i + CROP_DIM, j:j + CROP_DIM]
            pieces.append(croped_image)

    for idx in range(len(pieces)):
        piece = pieces[idx]
        np.save("{}_{}".format(saving_path, str(idx)), piece)
    return pieces


def create_binary_gt(input_matrix):
    binary_gt = np.zeros(input_matrix.shape, dtype=np.uint8)
    binary_gt[input_matrix == 2] = 1
    return binary_gt


def validate_arguments():
    arguments = args_parser()
    if not os.path.isdir(arguments.in_dir):
        raise RuntimeError("Input directory does not exist '%s'" % arguments.in_dir)
    if arguments.out_dir:
        if not os.path.isdir(arguments.out_dir):
            os.makedirs(arguments.out_dir)
    if arguments.train_dir:
        if not os.path.isdir(arguments.train_dir):
            os.makedirs(arguments.train_dir)
    if arguments.target_dir:
        if not os.path.isdir(arguments.target_dir):
            os.makedirs(arguments.target_dir)
    if arguments.xv_dir:
        if not os.path.isdir(arguments.xv_dir):
            os.makedirs(arguments.xv_dir)
    return arguments


def determine_sample_sizes(arguments, file_list):
    num_samples = len(file_list) / NUM_FILES_PER_SAMPLE
    num_xv_samples = 0
    num_training_samples = num_samples
    if arguments.xv_dir:
        training_factor = 1 - arguments.xv_sample_rate
        num_training_samples = training_factor * num_samples
        num_xv_samples = arguments.xv_sample_rate * num_samples

    return num_training_samples, num_xv_samples


def main():
    program_args = validate_arguments()
    files = os.listdir(program_args.in_dir)
    num_samples, num_xv = determine_sample_sizes(program_args, files)

    sites_tiles = {}
    sample_number = 0
    for file_name in files:
        site, tile, f_type, f_ext = split_file_name(file_name)

        current_key = site + "_" + tile
        if current_key not in sites_tiles.keys():
            sites_tiles[current_key] = {}

        if program_args.out_dir:
            types_to_look = ["DSM", "DTM", "RGB"]
        else:
            types_to_look = ["DSM", "DTM", "RGB", "GTL"]

        if f_type in types_to_look:
            if f_type == "DSM":
                processed_image = cv2.imread(os.path.join(program_args.in_dir, file_name), cv2.IMREAD_UNCHANGED)
            elif f_type == "DTM":
                processed_image = cv2.imread(os.path.join(program_args.in_dir, file_name), cv2.IMREAD_UNCHANGED)
            elif f_type == "RGB":
                processed_image = cv2.imread(os.path.join(program_args.in_dir, file_name), cv2.IMREAD_COLOR)
            elif f_type == "GTL":
                processed_image = cv2.imread(os.path.join(program_args.in_dir, file_name), cv2.IMREAD_GRAYSCALE)
            if sites_tiles[current_key] is not None:
                sites_tiles[current_key][f_type] = processed_image
                stored_images = sites_tiles[current_key]
                if len(stored_images.keys()) == len(types_to_look):
                    dsm_img = stored_images["DSM"]
                    dtm_img = stored_images["DTM"]
                    rgb_img = stored_images["RGB"]
                    saving_dir = program_args.train_dir
                    if not program_args.out_dir:
                        gt_image = stored_images["GTL"]
                        if sample_number <= num_samples:
                            saving_dir = program_args.train_dir
                        else:
                            saving_dir = program_args.xv_dir
                    else:
                        saving_dir = program_args.out_dir

                    height_matrix = dsm_img - dtm_img
                    final_image = np.dstack((rgb_img, height_matrix))
                    sample_number += 1
                    sites_tiles[current_key] = None
                    if program_args.target_dir:
                        crop_image(final_image, os.path.join(saving_dir, "{}_{}".format(site, tile)))
                        crop_image(create_binary_gt(gt_image), os.path.join(program_args.target_dir, "{}_{}".format(site, tile)))
                    if program_args.out_dir:
                        np.save(os.path.join(program_args.out_dir, "{}_{}".format(site, tile)), final_image)


if __name__ == '__main__':
    main()
