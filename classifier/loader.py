import os
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np

from segmentor.loader import RectangularPad


NORMALIZATION_COEFFICIENTS = {
    "mean": np.array([0.30847357, 0.31463413, 0.27157442]),
    "std": np.array([0.26294333, 0.26491422, 0.24728666]),
}


class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        from classifier import FOLDER_TO_TARGET

        self.root_dir = root_dir
        self.transform = transform

        self.list_path_images = []
        self.list_targets = []

        for class_directory in os.listdir(root_dir):
            if class_directory == "mistakes":
                continue
            for image_name in os.listdir(f"{root_dir}/{class_directory}"):
                self.list_path_images.append(f"{class_directory}/{image_name}")
                self.list_targets.append(FOLDER_TO_TARGET[class_directory])

    def __len__(self):
        return len(self.list_targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = pil_loader(f"{self.root_dir}/{self.list_path_images[idx]}")

        if self.transform:
            sample = self.transform(sample)

        return sample, self.list_targets[idx]


def get_transformation(input_size, data_augmentation):
    if data_augmentation:
        augmentation_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, hue=0.3)], p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.4),
            transforms.RandomApply([transforms.RandomRotation((-30, 30), expand=True)], p=0.7),
            transforms.RandomAdjustSharpness(4, p=0.2),
            transforms.RandomAutocontrast(p=0.7),
        ]
    else:
        augmentation_transform = []

    return transforms.Compose(
        augmentation_transform
        + [
            RectangularPad(input_size, input_size),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_COEFFICIENTS["mean"], NORMALIZATION_COEFFICIENTS["std"]),
        ]
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
    mean = NORMALIZATION_COEFFICIENTS["mean"]
    std = NORMALIZATION_COEFFICIENTS["std"]
    unnormalizer_operator = transforms.Normalize(-mean / std, 1 / std)

    return unnormalizer_operator(image)


def loader(path_data, input_size, split_type, batch_size, shuffle=True, data_augmentation=True):
    return torch.utils.data.DataLoader(
        CustomImageFolder(
            f"{path_data}/{split_type}_images", transform=get_transformation(input_size, data_augmentation)
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
