import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class RectangularPad:
    def __init__(self, input_height, input_width):
        self.input_height = input_height
        self.input_width = input_width

    def __call__(self, image):
        width, height = image.size[-2:]

        if height < width:
            total_padding = int(self.input_height * width / self.input_width) - height
            padding_height = total_padding // 2
            padding = (0, padding_height, 0, total_padding - padding_height)
        else:
            total_padding = int(self.input_width * height / self.input_height) - width
            padding_left = total_padding // 2
            padding = (padding_left, 0, total_padding - padding_left, 0)

        return F.pad(image, padding, 0, "constant")


class ToNumpy:
    def __call__(self, image):
        return np.array(image)


def get_transformation(input_height, input_width):
    return transforms.Compose(
        [RectangularPad(input_height, input_width), transforms.Resize((input_height, input_width)), ToNumpy()]
    )


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")
