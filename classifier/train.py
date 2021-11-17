import sys
import os

import torch


def train_cli(argvs=sys.argv[1:]):
    import argparse

    import torch.optim as optim

    from classifier.loader import loader
    from classifier.model import get_model
    from classifier.loss import cross_entropy_loss
    from classifier.validation import validation

    parser = argparse.ArgumentParser("Pipeline to train a model to classify the birds")
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
        "-pd",
        "--path_data",
        type=str,
        required=True,
        metavar="PD",
        help="the path that leads to the data, 'bird_dataset' will be added to the front (required)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, metavar="B", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument("-s", "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output",
        metavar="E",
        help="folder where experiment outputs are located, 'output' will be added to the front (default: output)",
    )
    args = parser.parse_args(argvs)
    print(args)

    args.output_path = "output/" + args.output_path
    # Create experiment folder
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # Torch meta settings
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Define the model, the loss and the optimizer
    model, input_size = get_model(args.model)
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    loss = cross_entropy_loss()

    # Define the data loaders
    args.path_data = "bird_dataset/" + args.path_data
    train_loader = loader(args.path_data, input_size, "train", args.batch_size, shuffle=True)
    validation_loader = loader(args.path_data, input_size, "val", args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.n_epochs + 1):
        print(f"Train Epoch {epoch}:")
        train_on_epoch(model, loss, optimizer, train_loader, use_cuda)
        validation(model, loss, validation_loader, use_cuda)

        if epoch % 2 == 1:
            weights_path = args.output_path + f"/{args.model}_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), weights_path)


def train_on_epoch(model, loss, optimizer, loader, use_cuda):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)
        loss_error = loss(output, target)

        loss_error.backward()
        optimizer.step()

        if batch_idx % (len(loader) // 10) == 0:
            print(
                f"[{batch_idx * len(data)}/{len(loader.dataset)} ({int(100.0 * batch_idx / len(loader))}%)]\tLoss: {loss_error.data.item():.6f}"
            )
