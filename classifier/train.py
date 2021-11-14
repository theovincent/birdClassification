import sys
import os

import torch
from tqdm import tqdm


def train_cli(argvs=sys.argv[1:]):
    import argparse

    import torch.optim as optim

    from classifier.loader import loader
    from classifier.model import Net
    from classifier.loss import cross_entropy_loss
    from classifier.validation import validation

    parser = argparse.ArgumentParser("Pipeline to train a model to classify the birds")
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64, metavar="B", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument("-m", "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("-s", "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output",
        metavar="E",
        help="folder where experiment outputs are located (default: output)",
    )
    args = parser.parse_args(argvs)
    print(args)

    # Create experiment folder
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # Torch meta settings
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Define the data loaders
    train_loader = loader("train", args.batch_size, shuffle=True)
    validation_loader = loader("val", args.batch_size, shuffle=False)

    # Define the model, the loss and the optimizer
    model = Net()
    if use_cuda:
        print("\n\n!! Using GPU !!\n\n")
        model.cuda()
    else:
        print("\n\n!! Using CPU !!\n\n")

    loss = cross_entropy_loss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(1, args.n_epochs + 1):
        print(f"Train Epoch {epoch}:")
        train_on_epoch(model, loss, optimizer, train_loader, use_cuda)
        validation(model, loss, validation_loader, use_cuda)

        weights_path = args.output_path + "/model_" + str(epoch) + ".pth"
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
