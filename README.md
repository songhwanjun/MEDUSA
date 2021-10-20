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

This project implements a framework to fuse RGB an depth information using multimodal Transformers in the context of object detection. The goal is to show that the inferred depth maps should play an important role to put the limit of appearance-based object detection. Thus, we propose a generic framework MEDUSA (Multimodal Estimated-Depth Unification with Self-Attention). Unlike previous methods that use the depth measured from various physical sensors such as Kinect and Lidar, we show that the depth maps inferred by a monocular depth estimator can play an important role to enhance the performance of modern object detectors. In order to make use of the estimated depth, MEDUSA encompasses a robust feature extraction phase, followed by multimodal transformers for RGB-D fusion, as can be seen in the figure above.


**What it is**. Unlike previous studies that use the depth measured from various
physical sensors such as Kinect and Lidar, MEDUSA is a novel object detection
pipeline that encompasses a robust feature extractor for RGB images and their
noisy (estimated) depth maps, followed by multimodal Transformers for achieving
optimal RGB-D fusion via the self-attention mechanism.

**About the code**. This code is based on the original implementation of DETR, a
transformer-based object detection framework, requiring Pytorch 1.5+.

## Datasets

We built two extended large-scale RGB-D datasets, COCO-RGBD and VOC-RGBD, by
applying the state-of-the-art monocular depth estimator, MiDaS. All the datasets
are provided with raw RGB images and their estimated depth maps, but we also
provide COCO-RGBD data with SSTable format for the testing purpose.

The two raw datasets can be found [COCO-RGBD](http://) and [VOC-RGBD](http://).
The COCO-RGBD (SSTable) can be found
[COCO-RGBD (SSTable)](https://cnsviewer.corp.google.com/cns/li-d/home/hwanjun/COCO2017-RGBD/TF/).
