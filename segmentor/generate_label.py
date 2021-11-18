import os

import numpy as np
import cv2
from tqdm import tqdm
import torch

from segmentor.loader import pil_loader, get_transformation
from segmentor.model import get_model


def generate_label():
    """Args:
    path_label
    path_to_label

    """
    args = {"path_label": "segmentation_from_network", "path_to_label": "train_images", "model": "detectron2"}

    path_to_label = f"bird_dataset/{args['path_to_label']}"
    path_label = f"bird_dataset/{args['path_label']}/{args['path_to_label']}"

    # Torch meta settings
    use_cuda = torch.cuda.is_available()

    # Define the model, the loss and the optimizer
    model, (input_height, input_width) = get_model(args["model"], use_cuda)
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    data_transformer = get_transformation(input_height, input_width)

    for folder in os.listdir(path_to_label):
        if not os.path.isdir(f"{path_label}/{folder}"):
            os.makedirs(f"{path_label}/{folder}")

        for file in tqdm(os.listdir(f"{path_to_label}/{folder}")):
            if "jpg" in file:
                image = data_transformer(pil_loader(f"{path_to_label}/{folder}/{file}"))

                cv2.imwrite(f"{path_label}/{folder}/{file[: -3]}_clean.png", image.astype(np.uint8))

                outputs = model(image)

                if 14 in outputs["instances"].__dict__["_fields"]["pred_classes"]:
                    index_bird = list(outputs["instances"].__dict__["_fields"]["pred_classes"]).index(14)

                    mask = np.array(outputs["instances"].__dict__["_fields"]["pred_masks"][index_bird]).reshape(
                        (input_height, input_width, 1)
                    )
                    label = np.repeat(mask, 3, axis=-1)

                    cv2.imwrite(f"{path_label}/{folder}/{file[: -3]}.png", label.astype(np.uint8) * 255)


if __name__ == "__main__":
    generate_label()
