from pathlib import Path
import torch
from torch.utils.data import Dataset
import json
import os, sys
from PIL import Image
import datasets.transforms_w_depth as T

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, image_size, transforms, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.data_folder = data_folder
        self.split = split.upper()
        self.image_size = image_size
        self._transforms = transforms

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):

        image_id = self.images[i]
        depth_id = os.path.join(self.data_folder, 'Depth', self.split, self.images[i].split('/')[-1])

        img = Image.open(image_id, mode='r').convert('RGB')
        depth = Image.open(depth_id, mode='r').convert('L')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = objects['boxes']
        labels = objects['labels']
        difficulties = objects['difficulties']

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            _boxes = []
            _labels = []
            _difficulties = []
            for id in range(len(difficulties)):
                if difficulties[id] == 0:
                    _boxes.append(boxes[id])
                    _labels.append(labels[id])
                    _difficulties.append(difficulties[id])
            boxes = _boxes
            labels = _labels
            difficulties = _difficulties

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        difficulties = torch.tensor(difficulties,dtype=torch.bool)

        target = {}
        str_file_name = self.images[i].split('/')[-1].split('.')[0]
        if len(str_file_name) > 6:
            file_id = int(str_file_name[0:4] + str_file_name[5:])
        else:
            file_id = int(str_file_name)

        # image, depth, target generate
        w, h = img.size
        file_id = torch.as_tensor([file_id])
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        target['boxes'] = boxes
        # -1 is needed: [0~19] object label, 20 backbround
        target['labels'] = labels
        target['image_id'] = file_id
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['difficulties'] = difficulties

        img, depth, target = self._transforms(img, depth, target)

        # reconstruct original depth map to be used as it is
        max = torch.max(depth[0])
        min = torch.min(depth[0])
        depth_map = (depth[0] - min) / (max - min)

        # rgb_feature, depth_feature, normalized depth_map, target
        return img, depth, depth_map, target

    def __len__(self):
        return len(self.images)

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
        if image_set == 'val' or image_set == 'test':
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

def build(image_set, args):
    root = Path(args.path)
    image_size = (args.image_height, args.image_width)

    if image_set == 'val':
        image_set = 'test'

    assert image_set == 'train' or image_set == 'test'

    dataset = PascalVOCDataset(root, image_set, image_size, transforms=make_coco_transforms(image_set, image_size))
    return dataset

