# TSTD

### Introduction
In this study, we present our novel TSTD model, which exhibits remarkable efficacy and efficiency, addressing the constraints encountered in prior research. Diverging from existing approaches that exclusively employ either geometry or RGB data for semantic segmentation, our proposed methodology incorporates both modalities within a unified, two-stage network architecture. This integrative approach enables the effective fusion of heterogeneous data features,
leading to notable enhancements in semantic segmentation outcomes. Moreover, we have devised an innovative and efficient decoder utilizing a lightweight transformer module


### The overall architecture of TSTD:
Network overview: our architecture consists of a 2D and a 3D components. The 2D component takes several RGB multi-view images as input, from
which it learns features. The output of the 2D network is the per-pixel uncertainty for each category. After the 3D encoder processes the input point cloud
with 2D semantic information, the extracted features are utilized in the 3D decoder with attention loss and segmentation loss.
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/architecture_7.0.png)


### A visual depiction of the original tokens and cross tokens:
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/tokens_3.0.png)


### The overall architecture of Trans-Decoder
The Trans-Decoder model comprises two pivotal components, namely original attention and cross attention. The attention loss L is designed to constrain on Xcross . ORE refers to the
original embedding operation, while CRE denotes the cross embedding operation.
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/trans-docoder_6.0.png)


### visualization results
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/vis.png)


# Installation
Our code is based on mmdetection3d. For install and data preparation, please refer to the guidelines in mmdetection3d https://github.com/open-mmlab/mmdetection3d.


# Training
Example:
```python
python tools\train.py configs\mve_seg\mve_seg_8x2_cosine_100e_scannet_seg-3d-23dim.py
```

# Results
### ScanNet
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/result.png)

