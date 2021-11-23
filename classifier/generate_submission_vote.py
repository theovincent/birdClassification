import sys
import os

from tqdm import tqdm

MODELS_TO_TEST = [
    {
        "model": "densenet",
        "classifier_4D": True,
        "path_weights": "segmentation_from_gt/densenet_41.pth",
        "path_to_test": "segmentation_from_network",
    },
    {
        "model": "resnet",
        "classifier_4D": True,
        "path_weights": "segmentation_from_gt/resnet_63.pth",
        "path_to_test": "segmentation_from_network",
    },
    {
        "model": "vgg",
        "classifier_4D": True,
        "path_weights": "segmentation_from_gt/vgg_33.pth",
        "path_to_test": "segmentation_from_network",
    },
    {
        "model": "vgg",
        "classifier_4D": False,
        "path_weights": "crop_from_gt/vgg_37.pth",
        "path_to_test": "crop_from_network",
    },
    {
        "model": "densenet",
        "classifier_4D": False,
        "path_weights": "crop_from_gt/densenet_21.pth",
        "path_to_test": "crop_from_network",
    },
    {
        "model": "resnet",
        "classifier_4D": False,
        "path_weights": "crop_from_gt/resnet_37.pth",
        "path_to_test": "crop_from_network",
    },
]


def generate_submission_vote_cli(argvs=sys.argv[1:]):
    import argparse

    import torch
    import pandas as pd
    import numpy as np

    from classifier.loader import pil_loader, get_geometric_transformation
    from classifier.model import get_model

    parser = argparse.ArgumentParser("Pipeline to generate a submision file for the Kaggle competition with .")
    parser.add_argument(
        "-ps",
        "--path_submission",
        type=str,
        required=True,
        metavar="PS",
        help="path where the submision csv file will to stored, 'output/submission/{args.path_submission}.csv' will be the final path (required)",
    )
    args = parser.parse_args(argvs)
    print(args)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        map_location = torch.device("cuda")
    else:
        print("\n\n!! Using CPU !!\n\n")
        map_location = torch.device("cpu")

    over_all_proba = pd.DataFrame(None, columns=[str(class_) for class_ in range(20)], dtype=float)
    softmax = torch.nn.Softmax(dim=None)

    for idx_model, model_cfg in tqdm(enumerate(MODELS_TO_TEST)):
        path_weights = f"output/{model_cfg['path_weights']}"
        path_to_test = f"bird_dataset/{model_cfg['path_to_test']}/test_images/mistery_category"

        # Retreive the model
        state_dict = torch.load(path_weights, map_location=map_location)
        model, input_size = get_model(model_cfg["model"], pretrained=False, classifier_4D=model_cfg["classifier_4D"])
        model.load_state_dict(state_dict)
        model.eval()
        if use_cuda:
            model.cuda()

        data_transformer = get_geometric_transformation(input_size, False, model_cfg["classifier_4D"])

        for image_name in os.listdir(path_to_test):
            if not model_cfg["classifier_4D"] and "jpg" in image_name:
                raw_image = pil_loader(f"{path_to_test}/{image_name}")
                data = data_transformer(raw_image)
                data = data.view(1, data.size(0), data.size(1), data.size(2))

                if use_cuda:
                    data = data.cuda()

                output = model(data)

                if idx_model == 0:
                    over_all_proba.loc[image_name[:-4]] = softmax(output)[0].detach().numpy()
                else:
                    over_all_proba.loc[image_name[:-4]] += softmax(output)[0].detach().numpy()

            elif model_cfg["classifier_4D"] and "png" in image_name:
                image_map = pil_loader(f"{path_to_test}/{image_name}")

                raw_image = pil_loader(
                    f"{path_to_test.replace(path_to_test.split('/')[1], 'raw_images')}/{image_name[:-3]}jpg"
                )

                data = torch.zeros([4] + list(raw_image.size[::-1]))
                data[:3] = torch.from_numpy(np.array(raw_image)).permute(2, 0, 1)
                data[3] = torch.from_numpy(np.array(image_map))[:, :, 0]

                data = data_transformer(data)
                data = data.view(1, data.size(0), data.size(1), data.size(2))

                if use_cuda:
                    data = data.cuda()

                output = model(data)

                if idx_model == 0:
                    over_all_proba.loc[image_name[:-4]] = softmax(output)[0].detach().numpy()
                else:
                    over_all_proba.loc[image_name[:-4]] += softmax(output)[0].detach().numpy()

    path_submission = f"output/submission/{args.path_submission}.csv"
    submission = open(path_submission, "w")
    submission.write("Id,Category\n")

    order_images = [
        image_path[:-4] for image_path in os.listdir(f"bird_dataset/raw_images/test_images/mistery_category")
    ]
    for image_name in over_all_proba.loc[order_images].index:
        pred = int(over_all_proba.loc[image_name].idxmax())

        submission.write("%s,%d\n" % (image_name, pred))

    submission.close()
