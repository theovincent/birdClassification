import sys
import os

from tqdm import tqdm


def generate_submission_cli(argvs=sys.argv[1:]):
    import argparse

    import torch

    from classifier.loader import pil_loader, data_transforms
    from classifier.model import Net

    parser = argparse.ArgumentParser("Pipeline to generate a submision file for the Kaggle competition.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        metavar="M",
        help="the model file to be evaluated. Usually it is of the form model_X.pth",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output/submission/kaggle.csv",
        metavar="D",
        help="name of the output csv file (output/submission/kaggle.csv)",
    )
    args = parser.parse_args(argvs)
    print(args)

    use_cuda = torch.cuda.is_available()

    # Retreive the model
    state_dict = torch.load("output/" + args.model)
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    output_file = open(args.output_path, "w")
    output_file.write("Id,Category\n")

    for file in tqdm(os.listdir("bird_dataset/test_images/mistery_category")):
        if "jpg" in file:
            data = data_transforms(pil_loader("bird_dataset/test_images/mistery_category/" + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            output_file.write("%s,%d\n" % (f[:-4], pred))

    output_file.close()

    print("Succesfully wrote " + args.output_path + ", you can upload this file to the kaggle competition website")
