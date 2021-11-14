import torch
from torchvision import datasets
import torchvision.transforms as transforms
import PIL.Image as Image

data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def loader(split_type, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(f"bird_dataset/{split_type}_images", transform=data_transforms),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )
