import sys

import numpy as np
from tqdm import tqdm

from classifier.loader import loader, unnormalizer


def compute_normalization_coefficients_cli(argvs=sys.argv[1:]):
    import argparse

    parser = argparse.ArgumentParser("Pipeline to get the normalization coefficients.")
    parser.add_argument(
        "-ptn",
        "--path_to_normalize",
        type=str,
        required=True,
        metavar="PTN",
        help="the path to the images we want to normalize, 'bird_dataset' will be added to the front (required)",
    )
    parser.add_argument(
        "-is",
        "--input_size",
        type=str,
        default=224,
        metavar="IS",
        help="the input size of the images to feed the neural network (default: 224)",
    )
    args = parser.parse_args(argvs)
    print(args)

    path_to_normalize = f"bird_dataset/{args.path_to_normalize}"

    data_loader = loader(path_to_normalize, args.input_size, "train", 1, shuffle=False, data_augmentation=False)

    mean_train_data = np.zeros(3)
    std_train_data = np.zeros(3)

    for image_batch, _ in tqdm(data_loader):
        image = np.array(unnormalizer(image_batch).permute(0, 2, 3, 1))[0]

        for channel in range(3):
            mean_train_data += np.mean(image[:, :, channel])
            std_train_data += np.std(image[:, :, channel])

    print("Mean per channel:\n", mean_train_data / len(data_loader))
    print("\nStd per channel:\n", std_train_data / len(data_loader))
