# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import math, os
import numpy as np
import torch, sys
import torch.utils.data
import torchvision
import random
import datasets.transforms_w_depth as T
from PIL import Image

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_set ,img_folder, ann_file, image_size, transforms, return_masks, subset_ratio=1.0):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.image_set = image_set
        self.image_size = image_size
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        # generate subset key to index mapping
        self.subset_index_dir = "subset_index"
        if not os.path.exists(self.subset_index_dir):
            os.mkdir(self.subset_index_dir)
        if image_set == 'train':
            self.key_to_index = self.generate_subset_mapping(subset_ratio)
        else:
            self.key_to_index = self.generate_subset_mapping(1.0)

    def __len__(self):
        return len(self.key_to_index)

    def __getitem__(self, key):

        # use mapping for subset
        idx = self.key_to_index[key]
        image_id = self.ids[idx]

        # get depthmap ####
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        img, target = super(CocoDetection, self).__getitem__(idx)
        depth = Image.open(os.path.join(str(self.root) + "_depth", file_name)).convert('L')

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # rgb_feature, gray_depth_feature, target
        img, depth, target = self._transforms(img, depth, target)

        # reconstruct original depth map to be used as it is
        max = torch.max(depth[0])
        min = torch.min(depth[0])
        depth_map = (depth[0] - min) / (max - min)

        # rgb_feature, depth_feature, normalized depth_map, target
        return img, depth, depth_map, target

    def generate_subset_mapping(self, subset_ratio):

        path = os.path.join(self.subset_index_dir, str(self.image_set) + "_" + str(subset_ratio))

        if os.path.exists(path):
            selected_ids = []
            f = open(path, 'r')
            for value in f:
                selected_ids.append(int(value.rstrip().strip()))
            selected_ids = np.array(selected_ids)

        else:
            total_num = len(self.ids)
            subset_num = int(math.ceil(total_num * subset_ratio))
            selected_ids = np.sort(np.random.choice(total_num, subset_num, replace=False))

            f = open(path, "w")
            for value in selected_ids:
                f.write(str(value) + "\n")

        select_map = {}
        for i in range(len(selected_ids)):
            select_map[i] = selected_ids[i]

        return select_map

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set, image_size):
    orig_height = 800
    orig_width = 1333
    orig_random_heights =[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    orig_crop_random_heights = [400, 500, 600]
    orig_crop_height = 384
    orig_crop_width = 600

    height = image_size[0]
    width = image_size[1]

    height_scale = float(height) / float(orig_height)
    width_scale = float(width) / float(orig_width)
    adj_height = int(orig_height * height_scale)
    adj_width = int(orig_width * width_scale)
    adj_random_heights = [int(scale*height_scale) for scale in orig_random_heights]
    adj_crop_random_heights = [int(scale * height_scale) for scale in orig_crop_random_heights]
    adj_crop_height = int(orig_crop_height * height_scale)
    adj_crop_width = int(orig_crop_width * width_scale)

    if image_size[0] < image_size[1]:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if image_set == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(adj_random_heights, max_size=adj_width),
                    T.Compose([
                        T.RandomResize(adj_crop_random_heights),
                        T.RandomSizeCrop(adj_crop_height, adj_crop_width),

                        T.RandomResize(adj_random_heights, max_size=adj_width),
                    ])
                ),
                normalize,
            ])
        if image_set == 'val':
            return T.Compose([
                T.RandomResize([adj_height], max_size=adj_width),
                normalize,
            ])
        raise ValueError(f'unknown {image_set}')
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return T.Compose([
            T.Resize((300,300)),
            normalize,
        ])

def make_coco_light_transforms(image_set, image_size):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        T.Resize((image_size[1], image_size[0])),
        normalize,
    ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.path)
    image_size = (args.image_height, args.image_width)
    subset_ratio = args.subset_ratio

    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    dataset = CocoDetection(image_set, img_folder, ann_file, image_size,
                            transforms=make_coco_transforms(image_set, image_size),
                            return_masks=False, subset_ratio=subset_ratio)
    return dataset
