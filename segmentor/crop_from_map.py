import os
import sys

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def crop_from_map_cli(argvs=sys.argv[1:]):
    import argparse

    parser = argparse.ArgumentParser("Pipeline to crop the images from map")
    parser.add_argument(
        "-ptc",
        "--path_to_crop",
        type=str,
        required=True,
        metavar="PTC",
        help="path to the images to segment, 'bird_dataset' will be added to the front (required)",
    )
    parser.add_argument(
        "-pm",
        "--path_maps",
        type=str,
        required=True,
        metavar="PM",
        help="path to the maps, 'bird_dataset/{args.path_maps}/{args.path_to_crop}' will be the final path (required)",
    )
    parser.add_argument(
        "-pc",
        "--path_crops",
        type=str,
        required=True,
        metavar="O",
        help="path where the cropped image will be stored, 'bird_dataset/{args.path_crops}/{args.path_to_crop}' will be the final path (required)",
    )
    args = parser.parse_args(argvs)
    print(args)

    path_to_crop = f"bird_dataset/{args.path_to_crop}"
    path_maps = f"bird_dataset/{args.path_maps}/{args.path_to_crop}"
    path_crops = f"bird_dataset/{args.path_crops}/{args.path_to_crop}"

    useful_folders = pd.Index(os.listdir(path_to_crop))[
        pd.Series(os.listdir(path_to_crop)).isin(pd.Series(os.listdir(path_maps)))
    ]

    for folder in tqdm(useful_folders):
        if folder == "false_negatives":
            continue

        path_to_image_folder = f"{path_to_crop}/{folder}"
        path_maps_folder = f"{path_maps}/{folder}"
        path_crops_folder = f"{path_crops}/{folder}"

        if not os.path.exists(path_crops_folder):
            os.makedirs(path_crops_folder)

        for name_image in os.listdir(path_to_image_folder):
            mask = cv2.imread(f"{path_maps_folder}/{name_image[:-3]}png")
            image = cv2.imread(f"{path_to_image_folder}/{name_image}")

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

            cv2.imwrite(f"{path_crops_folder}/{name_image}", cropped_image)
