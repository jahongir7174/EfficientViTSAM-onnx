import copy

import numpy
import torch
from onnxruntime import InferenceSession, SessionOptions


def get_image_size(input_image_size, longest_side):
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks, orig_im_size):
    img_size = 1024
    masks = torch.tensor(masks)
    orig_im_size = torch.tensor(orig_im_size)
    masks = torch.nn.functional.interpolate(masks,
                                            size=(img_size, img_size),
                                            mode="bilinear",
                                            align_corners=False, )

    pre_padded_size = get_image_size(orig_im_size, img_size)
    masks = masks[..., : int(pre_padded_size[0]), : int(pre_padded_size[1])]
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = torch.nn.functional.interpolate(masks,
                                            size=(h, w),
                                            mode="bilinear",
                                            align_corners=False)
    return masks


class Encoder:
    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = InferenceSession(model_path, SessionOptions(), providers=providers)
        self.inputs = self.session.get_inputs()[0].name

    def __call__(self, x, *args, **kwargs):
        return self.session.run(None, {self.inputs: x})[0]


class Decoder:
    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.target_size = 1024
        self.mask_threshold = 0.7
        self.session = InferenceSession(model_path, SessionOptions(), providers=providers)

    @staticmethod
    def get_preprocess_shape(old_h, old_w, long_side_length):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return list([new_h, new_w])

    def run(self,
            img_embeddings,
            origin_image_size,
            point_coords=None,
            point_labels=None,
            boxes=None):
        input_size = self.get_preprocess_shape(*origin_image_size, long_side_length=self.target_size)

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, input_size).astype(numpy.float32)

        if boxes is not None:
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).astype(numpy.float32)
            box_label = numpy.array([[2, 3] for _ in range(boxes.shape[0])], dtype=numpy.float32).reshape((-1, 2))
            point_coords = boxes
            point_labels = box_label

        input_dict = {"image_embeddings": img_embeddings, "point_coords": point_coords, "point_labels": point_labels}
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        masks = mask_postprocessing(low_res_masks, origin_image_size)

        masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    @staticmethod
    def apply_coords(coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = boxes.reshape(-1, 2, 2)
        return self.apply_coords(boxes, original_size, new_size)
