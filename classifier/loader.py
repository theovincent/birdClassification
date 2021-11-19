import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL.Image as Image


class SquarePad:
    def __call__(self, image):
        width, height = image.size[-2:]

        if height < width:
            padding_height = (width - height) // 2
            padding = (0, padding_height, 0, width - height - padding_height)
        else:
            padding_left = (height - width) // 2
            padding = (padding_left, 0, height - width - padding_left, 0)

        return F.pad(image, padding, 0, "constant")


def get_transformation(input_size):
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def loader(path_data, input_size, split_type, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(path_data + f"/{split_type}_images", transform=get_transformation(input_size)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )
