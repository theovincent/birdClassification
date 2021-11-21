import os
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np

from segmentor.loader import RectangularPad


NORMALIZATION_COEFFICIENTS = {"mean": [0.30847357, 0.31463413, 0.27157442], "std": [0.26294333, 0.26491422, 0.24728666]}


class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, path_data, split_type, color_transformation, geometric_transformation, classifier_4D=False):
        from classifier import FOLDER_TO_TARGET

        self.path_data = path_data
        self.split_type = split_type
        self.color_transform = color_transformation
        self.geometric_transformation = geometric_transformation
        self.classifier_4D = classifier_4D

        self.list_path_images = []
        self.list_targets = []

        if classifier_4D:
            self.root_dir = "bird_dataset/raw_images"
        else:
            self.root_dir = self.path_data

        for class_directory in os.listdir(f"{self.root_dir}/{self.split_type}_images"):
            if class_directory == "mistakes":
                continue
            for image_name in os.listdir(f"{self.root_dir}/{self.split_type}_images/{class_directory}"):
                self.list_path_images.append(f"{class_directory}/{image_name}")
                self.list_targets.append(FOLDER_TO_TARGET[class_directory])

        if classifier_4D:
            self.list_path_maps = [path_image.replace(".jpg", ".png") for path_image in self.list_path_images]

    def __len__(self):
        return len(self.list_targets)

    def __getitem__(self, idx):
        image = pil_loader(f"{self.root_dir}/{self.split_type}_images/{self.list_path_images[idx]}")

        if self.color_transform is not None:
            sample_ = self.color_transform(image)
        else:
            sample_ = image

        if self.classifier_4D:
            image_map = pil_loader(f"{self.path_data}/{self.split_type}_images/{self.list_path_maps[idx]}")

            sample = torch.zeros([4] + list(sample_.size[::-1]))
            sample[:3] = torch.from_numpy(np.array(sample_)).permute(2, 0, 1)
            sample[3] = torch.from_numpy(np.array(image_map))[:, :, 0]

            sample = self.geometric_transformation(sample)

        else:
            sample = self.geometric_transformation(sample_)

        return sample, self.list_targets[idx]


def get_color_transformation():
    return transforms.Compose(
        [
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5)], p=1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 4))], p=0.4),
            transforms.RandomAdjustSharpness(4, p=0.2),
            transforms.RandomAutocontrast(p=0.7),
        ]
    )


def get_geometric_transformation(input_size, data_augmentation, classifier_4D):
    if data_augmentation:
        augmentation_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation((-30, 30), expand=True)], p=0.7),
        ]
    else:
        augmentation_transform = []

    if classifier_4D:
        normalizer = [
            transforms.Normalize(NORMALIZATION_COEFFICIENTS["mean"] + [0], NORMALIZATION_COEFFICIENTS["std"] + [1])
        ]
    else:
        normalizer = [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_COEFFICIENTS["mean"], NORMALIZATION_COEFFICIENTS["std"]),
        ]

    return transforms.Compose(
        augmentation_transform + [RectangularPad(input_size, input_size), transforms.Resize(input_size)] + normalizer
    )


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def unnormalizer(image):
    """
    Normalization operation per channel: output = (input - mean) / std
    Unnormalization operation per chanel: input = (output - (-mean / std)) / std^-1
    """
    mean = np.array(NORMALIZATION_COEFFICIENTS["mean"])
    std = np.array(NORMALIZATION_COEFFICIENTS["std"])
    unnormalizer_operator = transforms.Normalize(-mean / std, 1 / std)

    return unnormalizer_operator(image)


def loader(path_data, input_size, split_type, batch_size, shuffle=True, data_augmentation=True, classifier_4D=False):
    return torch.utils.data.DataLoader(
        CustomImageFolder(
            path_data,
            split_type,
            get_color_transformation() if data_augmentation else None,
            get_geometric_transformation(input_size, data_augmentation, classifier_4D),
            classifier_4D=classifier_4D,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
