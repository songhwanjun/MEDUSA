from pathlib import Path
import torch
from torch.utils.data import Dataset
import os, sys
from PIL import Image
import datasets.transforms_w_depth as T
from scipy import io
import numpy as np
#import matplotlib.pyplot as plt

def box_cxcywh_to_xyxy(x):
    """Convert the coordinate format."""

    # [x_center, y_center, width, height]
    # -> [x1, y1, x2, y2]
    x_center, y_center, width, height = x.unbind(1)
    bbox = [(x_center - 0.5 * width), (y_center - 0.5 * height),
         (x_center + 0.5 * width), (y_center + 0.5 * height)]
    return torch.stack(bbox, dim=1)

def rescale_bboxes(out_bbox, size):
    """Resize the normalized coordinate into the original size."""
    img_width, img_height = size
    recaled_bboxes = box_cxcywh_to_xyxy(out_bbox)
    recaled_bboxes = recaled_bboxes * torch.tensor(
        [img_width, img_height, img_width, img_height],
        dtype=torch.float32)
    return recaled_bboxes

class SUNRGBDDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, image_size, transforms, gt_depth):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        self.data_folder = data_folder
        self.split = split.lower()
        self.image_size = image_size
        self._transforms = transforms

        # Multi-modal deep feature learning for RGB-D object detection (see)
        # .bfx indicates that the depth images obtained by running an inpainting algorithm on the raw depth map that contain
        # holes and missing vlaues to obtain a complete depth image for which each poixel has a depth value.
        self.voc_19_labels = (
        'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser', 'garbage_bin',
        'lamp', 'monitor', 'night_stand', 'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet')
        self.label_19_map = {k: v for v, k in enumerate(self.voc_19_labels)}
        self.rev_label_map = {v: k for k, v in self.label_19_map.items()}  # Inverse mapping

        assert self.split in {'train', 'test'}

        self.data_folder = data_folder
        self.rgb_folder = 'rgb'
        self.depth_folder = 'depth_gt' if gt_depth else 'depth_infer'

        # Read meta data
        self.annotations = self.get_annotations()
        self.file_names, self.file_keys = self.get_data_index()

    def __getitem__(self, i):

        image_id = self.file_names[i]
        image_key = self.file_keys[i]
        img = Image.open(os.path.join(self.data_folder, self.split, self.rgb_folder, image_id), mode='r').convert('RGB')
        depth = Image.open(os.path.join(self.data_folder, self.split, self.depth_folder, image_id), mode='r')

        objects = self.annotations[image_key]
        boxes = objects['boxes']
        labels = objects['labels']
        difficulties = objects['difficulties']
        etcs = objects['etc']

        # Excetion : no bbox
        if len(boxes) == 0:
            boxes = torch.tensor([[]], dtype=torch.float32)
            boxes = torch.reshape(boxes, (0, 4))
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        difficulties = torch.tensor(difficulties, dtype=torch.bool)

        target = {}
        file_id = int(image_id.split('.')[0])

        # image, depth, target generate
        w, h = img.size
        file_id = torch.as_tensor([file_id])
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        target['boxes'] = boxes
        # -1 is needed: [0~18] object label, 19 backbround
        target['labels'] = labels
        target['image_id'] = file_id
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['difficulties'] = difficulties

        # print(labels)
        img, depth, target = self._transforms(img, depth, target)

        # reconstruct original depth map to be used as it is
        max = torch.max(depth[0])
        min = torch.min(depth[0])
        depth_map = (depth[0] - min) / (max - min)

        # rgb_feature, depth_feature, normalized depth_map, target
        return img, depth, depth_map, target

    def __len__(self):
        return len(self.file_names)

    def get_annotations(self):

        mat = io.loadmat(os.path.join(self.data_folder, 'labels.mat'))
        meta = mat['SUNRGBDMeta2DBB'][0]

        annotations = {}
        for annotation in meta:

            key = '/n/fs/sun3d/data/' + annotation[0][0]
            # difficulties is a dummy structure to use the voc map-calculator, all values are filled with 0
            boxes, labels, etcs, difficulties = [], [], [], []
            _target = [] if len(annotation[1]) == 0 else annotation[1][0]

            for item in _target:

                box = item[1][0]
                box = box.astype(np.float32)
                # box format [x, y, w, h] -> [x, y, x, y]

                box[0] = float(box[0])
                box[1] = float(box[1])
                box[2] = float(box[0]) + float(box[2])
                box[3] = float(box[1]) + float(box[3])
                label = str(item[2][0])
                # For consistency with VOC
                etc = 1 - int(item[3][0])  # not, 0 means normal, 1 means difficult

                # filter only the labels in the 'voc_19_labels'
                if label in self.label_19_map:
                    boxes.append(box)
                    labels.append(self.label_19_map[label])
                    etcs.append(etc)
                    difficulties.append(0)

            # [x, y, w, h]
            boxes = np.asarray(boxes)
            boxes = boxes.astype(np.float32)
            labels = np.asarray(labels)
            etcs = np.asarray(etcs)
            difficulties = np.asarray(difficulties)
            annotations[key] = {
                'boxes': boxes, 'labels': labels, 'difficulties': difficulties, 'etc': etcs
            }

        return annotations

    def get_data_index(self):

        file_to_key = 'file_to_key.txt'

        # get all the file
        file_names = []
        file_keys = []

        f = open(os.path.join(self.data_folder, file_to_key), 'r')
        for line in f:
            tokens = line.rstrip().split('\t')
            file_name = tokens[0]
            key = tokens[1]

            if self.split in file_name:
                file_names.append(file_name.split('/')[1])
                file_keys.append(key)

        return file_names, file_keys

def make_coco_transforms(image_set, image_size):
    """A set of rules for data augmentation.

    Mostly copy-paste from https://github.com/facebookresearch/detr/
    blob/master/datasets/coco.py.
    The original code only supports 800x1333 resolution with the fixed rule.
    We extended the original code to support any resolution.

    Args:
        image_set: The subset type of data.
        image_size: The image size for resizing (data augmentation).

    Returns:

    Raises:
        ValueError: If the subset is not supported.
    """

    # The set of original rules for 800x1333 resolution.
    orig_height = 800
    orig_width = 1333
    orig_random_heights = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    orig_crop_random_heights = [400, 500, 600]
    orig_crop_height = 384
    orig_crop_width = 600

    # The target resolution for augmentation.
    height = image_size[0]
    width = image_size[1]

    # The modified rules for the target resolution.
    height_scale = float(height) / float(orig_height)
    width_scale = float(width) / float(orig_width)
    adj_height = int(orig_height * height_scale)
    adj_width = int(orig_width * width_scale)
    adj_random_heights = [
        int(scale * height_scale) for scale in orig_random_heights
    ]
    adj_crop_random_heights = [
        int(scale * height_scale) for scale in orig_crop_random_heights
    ]
    adj_crop_height = int(orig_crop_height * height_scale)
    adj_crop_width = int(orig_crop_width * width_scale)

    normalize = T.Compose(
        [T.ToTensor(),
         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(adj_random_heights, max_size=adj_width),
                T.Compose([
                    T.RandomResize(adj_crop_random_heights),
                    T.RandomSizeCrop(adj_crop_height, adj_crop_width),
                    T.RandomResize(adj_random_heights, max_size=adj_width),
                ])),
            normalize,
        ])
    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([adj_height], max_size=adj_width),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')



def build(image_set, args):

    root = Path(args.path)
    image_size = (args.image_height, args.image_width)
    gt_depth = args.gt_depth

    if image_set == 'val':
        image_set = 'test'

    assert image_set == 'train' or image_set == 'test'

    dataset = SUNRGBDDataset(root, image_set, image_size,
                             transforms=make_coco_transforms(image_set, image_size),
                             gt_depth=gt_depth)
    return dataset

