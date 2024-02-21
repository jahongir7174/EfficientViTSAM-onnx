import torch
from torchvision.transforms import functional


class Transform:
    def __init__(self, size):
        self.size = size

    def preprocess(self, x):
        pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

        x = torch.tensor(x)
        x = self.resize(x).float() / 255
        x = functional.normalize(x, mean=pixel_mean, std=pixel_std)

        h, w = x.shape[-2:]
        assert self.size >= h and self.size >= w
        x = torch.nn.functional.pad(x,
                                    (0, self.size - w, 0, self.size - h), value=0).unsqueeze(0).numpy()

        return x

    def resize(self, image):
        h, w, _ = image.shape
        if max(h, w) != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image):
        """
        Expects a torch tensor with shape HxWxC in float format.
        """
        shape = image.shape
        scale = self.size * 1.0 / max(shape[0], shape[1])

        new_h = shape[0] * scale
        new_h = int(new_h + 0.5)

        new_w = shape[1] * scale
        new_w = int(new_w + 0.5)
        return functional.resize(image.permute(2, 0, 1), list([new_h, new_w]))
