import sys
import os

import torch
import pandas as pd


def train_cli(argvs=sys.argv[1:]):
    import argparse

    import torch.optim as optim

    from classifier.loader import loader
    from classifier.model import get_model
    from classifier.loss import cross_entropy_loss
    from classifier.validation import validation

    parser = argparse.ArgumentParser("Pipeline to train a model to classify the birds")
    parser.add_argument(
        "-c",
        "--colab",
        default=False,
        action="store_true",
        help="if given, path_data will be modified with the correct path to the data in my google drive, otherwise nothing happens, (default: False)",
    )
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
        "-psw",
        "--path_starting_weights",
        type=str,
        default="ImageNet",
        metavar="PSW",
        help="the path to the starting weights, if None, takes random weights, 'output' will be added to the front (default: ImageNet)",
    )
    parser.add_argument(
        "-pd",
        "--path_data",
        type=str,
        required=True,
        metavar="PD",
        help="the path that leads to the data, 'bird_dataset' will be added to the front (required)",
    )
    parser.add_argument(
        "-nc",
        "--number_classes",
        type=int,
        default=20,
        metavar="NC",
        help="the number of classes to classify (default: 20)",
    )
    parser.add_argument(
        "-4D",
        "--classifier_4D",
        default=False,
        action="store_true",
        help="if given, a segmentation map will be added to the input, (default: False)",
    )
    parser.add_argument(
        "-fe",
        "--feature_extraction",
        default=False,
        action="store_true",
        help="if given, feature extraction will be performed, otherwise full training will be done, (default: False)",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=64, metavar="BS", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "-ne", "--n_epochs", type=int, default=1, metavar="NE", help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0005,
        metavar="LR",
        help="first learning rate before decreasing (default: 0.0005)",
    )
    parser.add_argument("-s", "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "-po",
        "--path_output",
        type=str,
        required=True,
        metavar="PO",
        help="folder where experiment outputs are located, 'output' will be added to the front (required)",
    )
    args = parser.parse_args(argvs)
    print(args)

    path_output = f"output/{args.path_output}"
    # Create experiment folder
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    # Torch meta settings
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")
    torch.manual_seed(args.seed)

    # Define the model, the loss and the optimizer
    if args.path_starting_weights is not None and args.path_starting_weights != "ImageNet":
        if args.colab:
            args.path_starting_weights = (
                f"/content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/{args.path_starting_weights}"
            )
        else:
            args.path_starting_weights = f"output/{args.path_starting_weights}"

    model, input_size = get_model(
        args.model,
        feature_extract=args.feature_extraction,
        path_starting_weights=args.path_starting_weights,
        num_classes=args.number_classes,
        classifier_4D=args.classifier_4D,
        map_location=map_location,
    )

    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    loss = cross_entropy_loss()
    losses = pd.DataFrame(
        None, index=range(1, args.n_epochs + 1), columns=["train_loss", "validation_loss", "validation_accuracy"]
    )

    # Define the data loaders
    if args.colab:
        args.path_data = (
            "/content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/bird_dataset/" + args.path_data
        )
    else:
        args.path_data = "bird_dataset/" + args.path_data
    train_loader = loader(
        args.path_data,
        input_size,
        "train",
        args.batch_size,
        shuffle=True,
        data_augmentation=True,
        classifier_4D=args.classifier_4D,
    )
    validation_loader = loader(
        args.path_data,
        input_size,
        "val",
        args.batch_size,
        shuffle=False,
        data_augmentation=False,
        classifier_4D=args.classifier_4D,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6, verbose=True)

    for epoch in range(1, args.n_epochs + 1):
        print(f"Train Epoch {epoch}:")
        train_loss = train_on_epoch(model, loss, optimizer, train_loader, use_cuda) / args.batch_size
        validation_loss, validation_accuracy = validation(model, loss, validation_loader, use_cuda)

        scheduler.step(validation_loss)

        losses.loc[epoch, ["train_loss", "validation_loss", "validation_accuracy"]] = [
            train_loss,
            validation_loss,
            validation_accuracy,
        ]
        if epoch % 2 == 1:
            path_weights = f"{path_output}/{args.model}_{str(epoch)}.pth"
            torch.save(model.state_dict(), path_weights)

        # Save at each epoch to be sure that the metrics are saved if an error occures
        losses.reset_index().to_feather(f"{path_output}/{args.model}.feather")


def train_on_epoch(model, loss, optimizer, loader, use_cuda):
    loss_on_batch = None
    model.train()

    for batch_idx, (data, target) in enumerate(loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)
        loss_error = loss(output, target)

        loss_error.backward()
        optimizer.step()

        if batch_idx % (len(loader) // 5) == 0:
            loss_on_batch = loss_error.data.item()
            print(
                f"[{batch_idx * len(data)}/{len(loader.dataset)} ({int(100.0 * batch_idx / len(loader))}%)]\tLoss: {loss_on_batch:.6f}"
            )

    return loss_on_batch
