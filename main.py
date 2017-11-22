import numpy as np
import argparse
import cv2


def create_binary_gt(input_matrix):
    binary_gt = np.ones(input_matrix.shape, dtype=np.uint8)*255
    binary_gt[input_matrix == 2] = 0
    return binary_gt


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsm", type=str, required=True, help="Digital Surface Model")
    parser.add_argument("--dtm", type=str, required=True, help="Digital Terrain Model")
    parser.add_argument("--rgb", type=str, required=True, help="RGB image file name")
    #parser.add_argument("--gtl", type=str, required=True, help="Ground truth levels")
    #parser.add_argument("--binary_gtl", type=str, required=True, help="File name for binary output")
    return parser

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    dsm_img = cv2.imread(args.dsm, cv2.IMREAD_UNCHANGED)
    dtm_img = cv2.imread(args.dtm, cv2.IMREAD_UNCHANGED)
    height_matrix = dsm_img - dtm_img
    rgb_img = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    print(rgb_img.shape)
    image = np.dstack((rgb_img, height_matrix))
    print(image.shape)