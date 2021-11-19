import os
import sys

import numpy as np
import cv2
from tqdm import tqdm
import torch

from segmentor.loader import pil_loader, get_transformation, get_inverse_transformation
from segmentor.model import get_model


def generate_segmentation_cli(argvs=sys.argv[1:]):
    import argparse

    parser = argparse.ArgumentParser("Pipeline to generate the labels from a network")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="M",
        help="the model name (required)",
        choices=["detectron2"],
    )
    parser.add_argument(
        "-o",
        "--path_to_segment",
        type=str,
        required=True,
        metavar="PTL",
        help="the path where the images to segment are, 'bird_dataset' will be added to the front (required)",
    )
    parser.add_argument(
        "-pd",
        "--path_maps",
        type=str,
        required=True,
        metavar="PL",
        help="the path where the maps with be stored, 'bird_dataset/{args.path_maps}/{args.path_to_segment}' will be the final path (required)",
    )
    args = parser.parse_args(argvs)
    print(args)

    path_to_segment = f"bird_dataset/{args.path_to_segment}"
    path_maps = f"bird_dataset/{args.path_maps}/{args.path_to_segment}"

    # Torch meta settings
    use_cuda = torch.cuda.is_available()

    # Define the model, the loss and the optimizer
    model, (input_height, input_width) = get_model(args.model, use_cuda)
    model.eval()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    data_transformer = get_transformation(input_height, input_width)

    if not os.path.isdir(f"{path_maps}/false_negatives"):
        os.makedirs(f"{path_maps}/false_negatives")

    for folder in os.listdir(path_to_segment):
        if not os.path.isdir(f"{path_maps}/{folder}"):
            os.makedirs(f"{path_maps}/{folder}")

        for file in tqdm(os.listdir(f"{path_to_segment}/{folder}")):
            if "jpg" in file:
                raw_image = pil_loader(f"{path_to_segment}/{folder}/{file}")
                (width, height) = raw_image.size
                image = data_transformer(raw_image)

                outputs = model(image)

                if 14 in outputs["instances"].__dict__["_fields"]["pred_classes"]:
                    index_bird = list(outputs["instances"].__dict__["_fields"]["pred_classes"]).index(14)

                    mask = np.array(outputs["instances"].__dict__["_fields"]["pred_masks"][index_bird]).reshape(
                        (input_height, input_width, 1)
                    )
                    label = np.repeat(mask, 3, axis=-1).astype(np.uint8) * 255

                    data_invertor = get_inverse_transformation(height, width)

                    cv2.imwrite(f"{path_maps}/{folder}/{file[: -3]}png", np.array(data_invertor(label)))
                else:
                    cv2.imwrite(f"{path_maps}/{folder}/{file[: -3]}png", np.ones(np.array(raw_image).shape) * 255)
                    cv2.imwrite(f"{path_maps}/false_negatives/{file[: -3]}png", np.array(raw_image))
