import argparse
from os import makedirs
from os.path import exists

import numpy
import yaml
from PIL import Image
from matplotlib import pyplot

from nets.nn import Decoder
from nets.nn import Encoder
from utils.util import Transform


def load_image(filename):
    with open(filename, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')
    return numpy.array(image)


def show_mask(mask, ax):
    color = numpy.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_mask(args, image, masks):
    pyplot.figure(figsize=(10, 10))
    pyplot.imshow(image)
    for mask in masks:
        show_mask(mask, pyplot.gca())
    pyplot.axis("off")
    pyplot.savefig(args.output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved in {args.output_path}")


def download(filename):
    import torch
    url = 'https://github.com/jahongir7174/EfficientViTSAM/releases/download/v0.0.1'
    if not exists(f'./weights/{filename}'):
        torch.hub.download_url_to_file(url=f'{url}/{filename}', dst=f'./weights/{filename}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="./demo/cat.jpg", type=str)
    parser.add_argument("--output_path", default="./demo/cat.png", type=str)
    parser.add_argument("--mode", default="point", choices=["point", "boxes"], type=str)
    parser.add_argument("--point", default=None, type=str)
    parser.add_argument("--boxes", default=None, type=str)
    args = parser.parse_args()

    if not exists('./weights'):
        makedirs('./weights')

    download('encoder_l0.onnx')
    download('encoder_l1.onnx')
    download('encoder_l2.onnx')
    download('decoder_l0.onnx')
    download('decoder_l1.onnx')
    download('decoder_l2.onnx')

    encoder = Encoder(model_path='./weights/encoder_l0.onnx')
    decoder = Decoder(model_path='./weights/decoder_l0.onnx')

    image = load_image(args.image_path)
    shape = image.shape[:2]

    transform = Transform(size=512)

    embeddings = encoder(transform.preprocess(image))

    if args.mode == "point":
        point = numpy.array(yaml.safe_load(args.point or f"[[[{shape[1] // 2}, {shape[0] // 2}, {1}]]]"),
                            dtype=numpy.float32)
        point_coords = point[..., :2]
        point_labels = point[..., 2]
        masks, _, _ = decoder.run(img_embeddings=embeddings,
                                  origin_image_size=shape,
                                  point_coords=point_coords,
                                  point_labels=point_labels)

        save_mask(args, image, masks)

    elif args.mode == "boxes":
        boxes = numpy.array(yaml.safe_load(args.boxes), dtype=numpy.float32)
        masks, _, _ = decoder.run(img_embeddings=embeddings,
                                  origin_image_size=shape,
                                  boxes=boxes)
        save_mask(args, image, masks)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
