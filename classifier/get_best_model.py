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
        choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "efficientnet"],
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

    odd_losses = losses.loc[losses.index % 2 == 1].copy()
    # Get the rank of the epoch weights according to the validation loss
    odd_losses["rank_loss"] = (
        losses.loc[losses.index % 2 == 1]
        .sort_values(by="validation_loss")
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
    )
    # Get the rank of the epoch weights according to the validation accuracy
    odd_losses["rank_accuracy"] = (
        losses.loc[losses.index % 2 == 1]
        .sort_values(by="validation_accuracy", ascending=False)
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
    )
    odd_losses["rank_sum"] = odd_losses["rank_loss"] + odd_losses["rank_accuracy"]
    odd_losses.sort_values(by="rank_sum").index[0]

    # Take the best summed rank
    print(int(odd_losses.sort_values(by="rank_sum").index[0]), end="")
