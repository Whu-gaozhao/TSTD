# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS
from mmdet3d.ops import PointFPModule


@NECKS.register_module()
class MveSegNECK(BaseModule):
    def __init__(self,
                 fp_channels=((768, 256, 256), (384, 256, 256),
                              (320, 256, 128), (128, 128, 128, 128)),
                #  fp_channels=((2560, 1536, 1024))
                 fp_norm_cfg=dict(type='BN2d'),
                 ignore_first_skip=True,
                 late_fuse=False,
                 init_cfg=None):
        super(MveSegNECK, self).__init__(init_cfg=init_cfg)

        self.num_fp = len(fp_channels)
        self.ignore_first_skip = ignore_first_skip
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            attention = True if late_fuse and cur_fp_mlps == fp_channels[
                -1] else False
            self.FP_modules.append(
                PointFPModule(
                    mlp_channels=cur_fp_mlps,
                    norm_cfg=fp_norm_cfg,
                    attention=attention))
    

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            list[torch.Tensor]: Coordinates of multiple levels of points.
            list[torch.Tensor]: Features of multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features

    
    def forward(self, feat_dict,pts_semantic_mask=None):
        sa_xyz, sa_features = self._extract_input(feat_dict)
        if self.ignore_first_skip:
            sa_features[0] = None

        fp_feature = sa_features[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)
        return fp_feature