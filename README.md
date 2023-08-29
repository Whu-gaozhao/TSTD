# TSTD

### The overall architecture of TSTD:
![image](https://github.com/Whu-gaozhao/TSTD/blob/master/resources/architecture_7.0.png)
Network overview: our architecture consists of a 2D and a 3D components. The 2D component takes several RGB multi-view images as input, from
which it learns features. The output of the 2D network is the per-pixel uncertainty for each category. After the 3D encoder processes the input point cloud
with 2D semantic information, the extracted features are utilized in the 3D decoder with attention loss and segmentation loss.
