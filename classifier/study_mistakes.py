import sys
import os

from tqdm import tqdm


def study_mistakes_cli(argvs=sys.argv[1:]):
    import argparse

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
        "-4D",
        "--classifier_4D",
        default=False,
        action="store_true",
        help="if given, a segmentation map will be added to the input, (default: False)",
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

    study_mistakes(args.model, args.classifier_4D, args.path_weights, args.path_to_evaluate, store=True)


def study_mistakes(args_model, args_classifier_4D, args_path_weights, args_path_to_evaluate, store=False):
    import torch
    import cv2
    import numpy as np
    import pandas as pd

    from classifier import FOLDER_TO_TARGET
    from classifier.loader import pil_loader, get_geometric_transformation
    from classifier.model import get_model

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    path_weights = f"output/{args_path_weights}"
    path_to_evaluate = f"bird_dataset/{args_path_to_evaluate}"
    path_store = f"bird_dataset/{args_path_to_evaluate}/mistakes/{args_path_weights}"

    # Retreive the model
    state_dict = torch.load(path_weights, map_location=map_location)
    model, input_size = get_model(args_model, pretrained=False, classifier_4D=args_classifier_4D)
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    data_transformer = get_geometric_transformation(input_size, False, args_classifier_4D)

    if not os.path.exists(path_store) and store:
        os.makedirs(path_store)

    proba_answers = pd.DataFrame(columns=range(20))
    targets = pd.Series(name="target")
    softmax = torch.nn.Softmax(dim=None)

    for folder in tqdm(os.listdir(path_to_evaluate)):
        if folder == "mistakes":
            continue

        path_to_evaluate_folder = f"{path_to_evaluate}/{folder}"

        for image_name in os.listdir(path_to_evaluate_folder):
            if not args_classifier_4D and "jpg" in image_name:
                raw_image = pil_loader(f"{path_to_evaluate_folder}/{image_name}")
                data = data_transformer(raw_image)
                data = data.view(1, data.size(0), data.size(1), data.size(2))

                if use_cuda:
                    data = data.cuda()

                output = model(data)
                pred = int(output.data.max(1, keepdim=True)[1])
                correct = pred == FOLDER_TO_TARGET[folder]

                proba_answers.loc[image_name[:-4]] = softmax(output)[0].detach().numpy()
                targets[image_name[:-4]] = FOLDER_TO_TARGET[folder]

                if not correct and store:
                    cv2.imwrite(f"{path_store}/{image_name}", np.array(raw_image))

            elif args_classifier_4D and "png" in image_name:
                image_map = pil_loader(f"{path_to_evaluate_folder}/{image_name}")

                raw_image = pil_loader(
                    f"{path_to_evaluate.replace(path_to_evaluate.split('/')[1], 'raw_images')}/{folder}/{image_name[:-3]}jpg"
                )

                data = torch.zeros([4] + list(raw_image.size[::-1]))
                data[:3] = torch.from_numpy(np.array(raw_image)).permute(2, 0, 1)
                data[3] = torch.from_numpy(np.array(image_map))[:, :, 0]

                data = data_transformer(data)
                data = data.view(1, data.size(0), data.size(1), data.size(2))

                if use_cuda:
                    data = data.cuda()

                output = model(data)
                pred = int(output.data.max(1, keepdim=True)[1])
                correct = pred == FOLDER_TO_TARGET[folder]

                proba_answers.loc[image_name[:-4]] = softmax(output)[0].detach().numpy()
                targets[image_name[:-4]] = FOLDER_TO_TARGET[folder]

                if not correct and store:
                    cv2.imwrite(f"{path_store}/{image_name[:-3]}jpg", np.array(raw_image))

    return proba_answers, targets
