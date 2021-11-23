import sys
import os

from tqdm import tqdm


def generate_submission_cli(argvs=sys.argv[1:]):
    import argparse

    import torch
    import numpy as np

    from classifier.loader import pil_loader, get_geometric_transformation
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
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    path_weights = f"output/{args.path_weights}"
    path_to_test = f"bird_dataset/{args.path_to_test}/test_images/mistery_category"
    path_submission = f"output/submission/{args.path_submission}.csv"

    # Retreive the model
    state_dict = torch.load(path_weights, map_location=map_location)
    model, input_size = get_model(args.model, pretrained=False, classifier_4D=args.classifier_4D)
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    data_transformer = get_geometric_transformation(input_size, False, args.classifier_4D)

    submission = open(path_submission, "w")
    submission.write("Id,Category\n")

    for image_name in tqdm(os.listdir(path_to_test)):
        if not args.classifier_4D and "jpg" in image_name:
            raw_image = pil_loader(f"{path_to_test}/{image_name}")
            data = data_transformer(raw_image)
            data = data.view(1, data.size(0), data.size(1), data.size(2))

            if use_cuda:
                data = data.cuda()

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            submission.write("%s,%d\n" % (image_name[:-4], pred))

        elif args.classifier_4D and "png" in image_name:
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
            pred = output.data.max(1, keepdim=True)[1]

            submission.write("%s,%d\n" % (image_name[:-4], pred))

    submission.close()
