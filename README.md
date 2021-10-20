# Exploiting Scene Depth for Object Detection with Multimodal Transformers

* **`Oct 17, 2021`:** **Our work is accepted at BMVC 2021**

## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@inproceedings{MEDUSA,
  title={Exploiting Scene Depth for Object Detection with Multimodal Transformers},
  author={Song, Hwanjun and Kim, Eunyoung and Jampani, Varun and Sun, Deqing and Lee, Jae-Gil and Yang, Ming-Hsuan},
  booktitle={BMVC},
  year={2021}
}
```

## Overview

<p align="center">
<img src="featured.png " width="1200"> 
</p>

This project implements a framework to fuse RGB an depth information using multimodal Transformers in the context of object detection. The goal is to show that the inferred depth maps should play an important role to put the limit of appearance-based object detection. Thus, we propose a generic framework MEDUSA (Multimodal Estimated-Depth Unification with Self-Attention). Unlike previous methods that use the depth measured from various physical sensors such as Kinect and Lidar, we show that the depth maps inferred by a monocular depth estimator can play an important role to enhance the performance of modern object detectors. In order to make use of the estimated depth, MEDUSA encompasses a robust feature extraction phase, followed by multimodal transformers for RGB-D fusion, as can be seen in the figure above.


**What it is**. Unlike previous studies that use the depth measured from various
physical sensors such as Kinect and Lidar, MEDUSA is a novel object detection
pipeline that encompasses a robust feature extractor for RGB images and their
noisy (estimated) depth maps, followed by multimodal Transformers for achieving
optimal RGB-D fusion via the self-attention mechanism.

**About the code**. This code is based on the original implementation of DETR, a
transformer-based object detection framework, requiring Pytorch 1.5+.

## Microsoft COCO Data with Inferred Depth Maps

We extended a large-scale objecct detection dataset, Microsoft COCO, by applying the state-of-the-art monocular depth estimator, [MiDaS](https://github.com/isl-org/MiDaS). The extracted depth maps for Microsoft COCO is available [[here]](https://drive.google.com/file/d/1TyXIqykl_T6SmDBZJ0-y6OYnKVKRv9Aq/view?usp=sharing). Please put the two folders, train2017_depth and val2017_depth, at the same location where train_2017 and val2017 folders exist.

## Training
Please run the run_distributed_medusa.py file using suitable hyperparameters inside. 
```
# Hyperparameters in run_distributed_medusa.py (e.g., 8 GPU setup).
batch_size = '4'
image_height = 800
image_width = 1333
num_workers = '2'
epochs = '150'
lr_drop = 100
print_freq = 200
path = '/home/Research/COCO2017'

# run the training script.
python run_distributed_medusa.py
```

## Comparison with RGB-Only DETR

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>resolution</th>
      <th>schedule</th>
      <th>box AP</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>DETR</td>
      <td>300x500</td>
      <td>150</td>
      <td>27.5</td>
      <td>-</td>
    <tr>
      <th>3</th>
      <td>DETR</td>
      <td>420x700</td>
      <td>150</td>
      <td>32.5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DETR</td>
      <td>800x1333</td>
      <td>150</td>
      <td>38.3</td>
      <td>-</td>
    </tr>
    </tr>
    <tr>
      <th>2</th>
      <td>MEDUSA</td>
      <td>300x500</td>
      <td>150</td>
      <td>28.9 (+1.4)</td>
      <td><a href="https://drive.google.com/file/d/1BYtXs6VZJA4VHx0ozW8GxzJ3dZYJ16yR/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/170lin-9qhuggYyqeTbwLHawFuQDL5rrt/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <th>4</th>
      <td>MEDUSA</td>
      <td>420x700</td>
      <td>150</td>
      <td>33.6 (+1.1)</td>
      <td><a href="https://drive.google.com/file/d/1RdyySEw4hmjZY0BgPfWkRdxf15iRMAHH/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1nCBKNFM_Jp-8ApmIqJlp3LDfyw7cB-m6/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <th>6</th>
      <td>MEDUSA</td>
      <td>800x1333</td>
      <td>150</td>
      <td>40.0 (+1.7)</td>
      <td><a href="https://drive.google.com/file/d/1jhxyNOLngHyvvcjfXT-Osi-RuBGh2qbS/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1-MppNO-I0nDvJyivpsA6vjAdYCDxlKgZ/view?usp=sharing">logs</a></td>
    </tr>
  </tbody>
</table>

