import sys
import os

from tqdm import tqdm


def generate_submission_cli(argvs=sys.argv[1:]):
    import argparse

    import torch

    from classifier.loader import pil_loader, get_transformation
    from classifier.model import get_model

    parser = argparse.ArgumentParser("Pipeline to generate a submision file for the Kaggle competition.")
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
        "-ptt",
        "--path_to_test",
        type=str,
        required=True,
        metavar="PTT",
        help="the path to the test images, 'bird_dataset/{args.path_to_test}/test_images/mistery_category' will be the final path (required)",
    )
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
        map_location = torch.device("gpu")
    else:
        map_location = torch.device("cpu")

    path_weights = f"output/{args.path_weights}"
    path_to_test = f"bird_dataset/{args.path_to_test}/test_images/mistery_category"
    path_submission = f"output/submission/{args.path_submission}.csv"

    # Retreive the model
    state_dict = torch.load(path_weights, map_location=map_location)
    model, input_size = get_model(args.model)
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    data_transformer = get_transformation(input_size, False)

    submission = open(path_submission, "w")
    submission.write("Id,Category\n")

    for image_name in tqdm(os.listdir(path_to_test)):
        if "jpg" in image_name:
            data = data_transformer(pil_loader(f"{path_to_test}/{image_name}"))
            data = data.view(1, data.size(0), data.size(1), data.size(2))

            if use_cuda:
                data = data.cuda()

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            submission.write("%s,%d\n" % (image_name[:-4], pred))

    submission.close()
