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

        total_padding_height = int(self.input_height * width / self.input_width) - height
        total_padding_width = int(self.input_width * height / self.input_height) - width

        if total_padding_height >= 0:
            padding_top = total_padding_height // 2
            padding = (0, padding_top, 0, total_padding_height - padding_top)
        else:
            padding_left = total_padding_width // 2
            padding = (padding_left, 0, total_padding_width - padding_left, 0)

        return F.pad(image, padding, 0, "constant")


class RectangularCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        scaled_width, scaled_height = image.size[-2:]

        if scaled_width == self.width >= 0:
            # Height was padded
            total_padding = scaled_height - self.height
            padding_top = total_padding // 2
            padding_bottom = total_padding - padding_top
            # (left, top, right, bottom)
            crop_size = (0, padding_top, self.width, scaled_height - padding_bottom)

        else:
            total_padding = scaled_width - self.width
            padding_left = total_padding // 2
            padding_right = total_padding - padding_left
            crop_size = (padding_left, 0, scaled_width - padding_right, self.height)

        return image.crop(crop_size)


class ScaleResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        input_width, input_height = image.size[-2:]

        total_padding_height = int(input_height * self.width / input_width) - self.height
        total_padding_width = int(input_width * self.height / input_height) - self.width

        if total_padding_height >= 0:
            # Height was padded
            padded_height = total_padding_height + self.height
            scaled_size = (padded_height, self.width)
        else:
            padded_width = total_padding_width + self.width
            scaled_size = (self.height, padded_width)

        return F.resize(image, scaled_size)


class ToNumpy:
    def __call__(self, image):
        return np.array(image)


class ToPIL:
    def __call__(self, image):
        return Image.fromarray(image)


def get_transformation(input_height, input_width):
    return transforms.Compose(
        [RectangularPad(input_height, input_width), transforms.Resize((input_height, input_width)), ToNumpy()]
    )


def get_inverse_transformation(height, width):
    return transforms.Compose([ToPIL(), ScaleResize(height, width), RectangularCrop(height, width)])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")
