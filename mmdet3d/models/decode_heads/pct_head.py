from mmcv.cnn.bricks import ConvModule

import torch
from torch import nn as nn
from torch.nn import functional as F

from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead


@HEADS.register_module()
class Point_TransformerHead(Base3DDecodeHead):
    r"""Point_Transformer decoder head.
    Two difference between this and the original:
        no cls_label
        location of DropOut is before the last conv
    Args:

    """

    def __init__(self,
                 in_channels=1024,
                 mid_channels=512,
                 **kwargs):
        super(Point_TransformerHead, self).__init__(**kwargs)

        self.conv = ConvModule(
            3 * in_channels,
            mid_channels,
            1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pre_seg_conv = ConvModule(
            mid_channels,
            self.channels,
            1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x):
        """

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """
        B, _, N = x.shape
        x_max = F.adaptive_max_pool1d(x, 1).view(B, -1).unsqueeze(-1).repeat(
            1, 1, N)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1).unsqueeze(-1).repeat(
            1, 1, N)
        x = torch.cat([x, x_max, x_avg], dim=1)

        output = self.conv(x)
        output = self.pre_seg_conv(output)
        output = self.cls_seg(output)

        return output