import os
import sys

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def segment_from_label_cli(argvs=sys.argv[1:]):
    import argparse

    parser = argparse.ArgumentParser("Pipeline to crop the images from some labels")
    parser.add_argument(
        "-pts",
        "--path_to_segment",
        type=str,
        default="bird_dataset/train_images",
        metavar="PTS",
        help="path to the images to segment (default: 'bird_dataset/train_images')",
    )
    parser.add_argument(
        "-ptl",
        "--path_labels",
        type=str,
        default="bird_dataset/segmentations",
        metavar="PL",
        help="path to the labels, should be mask of the birds (default: 'bird_dataset/segmentations')",
    )
    parser.add_argument(
        "-o",
        "--path_output",
        type=str,
        required=True,
        metavar="O",
        help="path to the output where the cropped image will be (required)",
    )
    args = parser.parse_args(argvs)
    print(args)

    segment_from_label(args.path_to_segment, args.path_labels, args.path_output)


def segment_from_label(path_to_segment, path_labels, path_output):
    useful_folders = pd.Index(os.listdir(path_to_segment))[
        pd.Series(os.listdir(path_to_segment)).isin(pd.Series(os.listdir(path_labels)))
    ]

    for folder in tqdm(useful_folders):
        path_to_image_folder = path_to_segment + "/" + folder
        path_to_labels_folder = path_labels + "/" + folder
        path_output_folder = path_output + "/" + folder

        if not os.path.exists(path_output_folder):
            os.makedirs(path_output_folder)

        for name_image in os.listdir(path_to_image_folder):
            mask = cv2.imread(path_to_labels_folder + "/" + name_image[:-3] + "png")
            image = cv2.imread(path_to_image_folder + "/" + name_image)

            # Get the borders
            max_along_axis_1 = pd.Series(np.max(mask, axis=(1, 2)))
            max_along_axis_0 = pd.Series(np.max(mask, axis=(0, 2)))

            mask_along_axis_1 = max_along_axis_1[max_along_axis_1 > 0].index
            mask_along_axis_0 = max_along_axis_0[max_along_axis_0 > 0].index

            top_border = mask_along_axis_1.min()
            bottom_border = mask_along_axis_1.max()
            left_border = mask_along_axis_0.min()
            right_border = mask_along_axis_0.max()

            cropped_image = image[top_border : bottom_border + 1, left_border : right_border + 1].copy()

            cv2.imwrite(path_output_folder + "/" + name_image, cropped_image)
