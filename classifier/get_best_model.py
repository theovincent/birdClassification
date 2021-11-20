import sys


def get_best_model_cli(argvs=sys.argv[1:]):
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser("Pipeline to get the best model at the end of a training.")
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
        "-po",
        "--path_output",
        type=str,
        required=True,
        metavar="PO",
        help="folder where experiment outputs are located, 'output' will be added to the front (required)",
    )
    args = parser.parse_args(argvs)

    losses = pd.read_feather(f"output/{args.path_output}/{args.model}.feather").set_index("index")

    print(int(losses.loc[losses.index % 2 == 1, "validation_accuracy"].idxmax()), end="")
