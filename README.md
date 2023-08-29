# TSTD

### The overall architecture of TSTD:
![image](https://github.com/Whu-gaozhao/TSTD/blob/master/resources/architecture_7.0.png)
Network overview: our architecture consists of a 2D and a 3D components. The 2D component takes several RGB multi-view images as input, from
which it learns features. The output of the 2D network is the per-pixel uncertainty for each category. After the 3D encoder processes the input point cloud
with 2D semantic information, the extracted features are utilized in the 3D decoder with attention loss and segmentation loss.

### A visual depiction of the original tokens and cross tokens:
<div align=center>
<img src="[https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-2.png](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/tokens_3.0.png)https://github.com/Whu-gaozhao/TSTD/blob/main/resources/tokens_3.0.png" width="180" height="105"> width="180" height="105"/>
</div>
OIT means original information tokens.CIT means cross information tokens.



