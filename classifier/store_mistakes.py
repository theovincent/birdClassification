import sys
import os

from tqdm import tqdm


def store_mistakes_cli(argvs=sys.argv[1:]):
    import argparse

    import torch
    import cv2
    import numpy as np

    from classifier import TARGET_TO_FOLDER
    from classifier.loader import pil_loader, get_transformation
    from classifier.model import get_model

    parser = argparse.ArgumentParser("Pipeline to store the clissification mistakes done by a neural network.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="M",
        help="the model name (required)",
        choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"],
    )
    parser.add_argument(
        "-pw",
        "--path_weights",
        type=str,
        required=True,
        metavar="PW",
        help="the path to the weights, 'output' will be added to the front (required)",
    )
    parser.add_argument(
        "-pte",
        "--path_to_evaluate",
        type=str,
        required=True,
        metavar="PTE",
        help="the path to the images we want to store the mistakes, 'bird_dataset' will be added to the front (required)",
    )
    args = parser.parse_args(argvs)
    print(args)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    path_weights = f"output/{args.path_weights}"
    path_to_evaluate = f"bird_dataset/{args.path_to_evaluate}"
    path_store = f"bird_dataset/{args.path_to_evaluate}/mistakes/{args.path_weights}"

    # Retreive the model
    state_dict = torch.load(path_weights, map_location=map_location)
    model, input_size = get_model(args.model, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    data_transformer = get_transformation(input_size)

    if not os.path.exists(path_store):
        os.makedirs(path_store)

    for folder in os.listdir(path_to_evaluate):
        if folder == "mistakes":
            continue

        path_to_evaluate_folder = f"{path_to_evaluate}/{folder}"

        for image_name in tqdm(os.listdir(path_to_evaluate_folder)):
            if "jpg" in image_name:
                raw_image = pil_loader(f"{path_to_evaluate_folder}/{image_name}")
                data = data_transformer(raw_image)
                data = data.view(1, data.size(0), data.size(1), data.size(2))

                if use_cuda:
                    data = data.cuda()

                output = model(data)
                pred = int(output.data.max(1, keepdim=True)[1])
                correct = TARGET_TO_FOLDER[pred] == folder

                if not correct:
                    cv2.imwrite(f"{path_store}/{image_name[: -3]}png", np.array(raw_image))
