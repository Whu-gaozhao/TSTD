# TSTD

### The overall architecture of TSTD:
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/architecture_7.0.png)
Network overview: our architecture consists of a 2D and a 3D components. The 2D component takes several RGB multi-view images as input, from
which it learns features. The output of the 2D network is the per-pixel uncertainty for each category. After the 3D encoder processes the input point cloud
with 2D semantic information, the extracted features are utilized in the 3D decoder with attention loss and segmentation loss.

### A visual depiction of the original tokens and cross tokens:
<div align=center>
<img src="https://github.com/Whu-gaozhao/TSTD/blob/main/resources/tokens_3.0.png" 
</div>

### The overall architecture of Trans-Decoder
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/trans-docoder_6.0.png)
The Trans-Decoder model comprises two pivotal components, namely original attention and cross attention. The attention loss L is designed to constrain on Xcross . ORE refers to the
original embedding operation, while CRE denotes the cross embedding operation.

### visualization results
![image](https://github.com/Whu-gaozhao/TSTD/blob/main/resources/vis.png)



