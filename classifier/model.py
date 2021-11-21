import torch.nn as nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_model(model_name, feature_extract=False, pretrained=True, num_classes=20, classifier_4D=False):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        if classifier_4D:
            first_out_channels = model_ft.conv1.out_channels
            first_pretrained_weights = model_ft.conv1.weight.data

            model_ft.conv1 = nn.Conv2d(
                4, first_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            model_ft.conv1.weight.data[:, :3] = first_pretrained_weights

    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        if classifier_4D:
            list_first_sequence = list(model_ft.features.children())
            first_out_channels = list_first_sequence[0].out_channels
            first_pretrained_weights = list_first_sequence[0].weight.data
            first_pretrained_bias = list_first_sequence[0].bias.data

            list_first_sequence[0] = nn.Conv2d(
                4, first_out_channels, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)
            )
            list_first_sequence[0].weight.data[:, :3] = first_pretrained_weights
            list_first_sequence[0].bias.data = first_pretrained_bias

            model_ft.features = nn.Sequential(*list_first_sequence)

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        if classifier_4D:
            list_first_sequence = list(model_ft.features.children())
            first_out_channels = list_first_sequence[0].out_channels
            first_pretrained_weights = list_first_sequence[0].weight.data
            first_pretrained_bias = list_first_sequence[0].bias.data

            list_first_sequence[0] = nn.Conv2d(4, first_out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            list_first_sequence[0].weight.data[:, :3] = first_pretrained_weights
            list_first_sequence[0].bias.data = first_pretrained_bias

            model_ft.features = nn.Sequential(*list_first_sequence)

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

        if classifier_4D:
            list_first_sequence = list(model_ft.features.children())
            first_out_channels = list_first_sequence[0].out_channels
            first_pretrained_weights = list_first_sequence[0].weight.data
            first_pretrained_bias = list_first_sequence[0].bias.data

            list_first_sequence[0] = nn.Conv2d(4, first_out_channels, kernel_size=(7, 7), stride=(2, 2))
            list_first_sequence[0].weight.data[:, :3] = first_pretrained_weights
            list_first_sequence[0].bias.data = first_pretrained_bias

            model_ft.features = nn.Sequential(*list_first_sequence)

    elif model_name == "densenet":
        """Densenet"""
        model_ft = models.densenet121(pretrained=pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        if classifier_4D:
            list_first_sequence = list(model_ft.features.children())
            first_out_channels = list_first_sequence[0].out_channels
            first_pretrained_weights = list_first_sequence[0].weight.data

            list_first_sequence[0] = nn.Conv2d(
                4, first_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            list_first_sequence[0].weight.data[:, :3] = first_pretrained_weights

            model_ft.features = nn.Sequential(*list_first_sequence)

    elif model_name == "efficientnet":
        """efficientnet"""
        model_ft = models.efficientnet_b7(pretrained=pretrained)
        num_ftrs = 2560
        model_ft.classifier = nn.Sequential(nn.Dropout(), nn.Linear(num_ftrs, num_classes))
        input_size = 224

        if classifier_4D:
            list_first_sequence = list(model_ft.features.children())
            list_first_conv = list(list_first_sequence[0].children())
            first_out_channels = list_first_conv[0].out_channels
            first_pretrained_weights = list_first_conv[0].weight.data

            list_first_conv[0] = nn.Conv2d(
                4, first_out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            list_first_conv[0].weight.data[:, :3] = first_pretrained_weights
            list_first_sequence[0] = nn.Sequential(*list_first_conv)
            model_ft.features = nn.Sequential(*list_first_sequence)

    else:
        print("Invalid model name, exiting...")
        exit()

    set_parameter_requires_grad(model_ft, feature_extract)

    return model_ft, input_size
