# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.ops import (GroupAll, PAConv, Points_Sampler, QueryAndGroup,
                         gather_points)
from .builder import SA_MODULES
from .point_sa_module import BasePointSAModule


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x = self.attn(self.norm1(x), identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        return x


@SA_MODULES.register_module()
class TransformerSAModule(BasePointSAModule):

    def __init__(self,
                 num_point,
                 radius,
                 num_sample,
                 mlp_channels,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(TransformerSAModule, self).__init__(
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            mlp_channels=[mlp_channels],
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        assert len(mlp_channels) == 4

        if use_xyz:
            mlp_channels[0] = mlp_channels[0] + 3
        self.mlp = ConvModule(
            mlp_channels[0],
            mlp_channels[1],
            kernel_size=(1, 1),
            stride=(1, 1),
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            bias=bias)
        self.transformer_layer = TransformerEncoderLayer(
            mlp_channels[1] * num_sample, mlp_channels[1], 2 * num_sample)
        self.after_mlp = ConvModule(
            mlp_channels[1] * num_sample,
            mlp_channels[-1],
            kernel_size=1,
            stride=1,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            bias=bias)

    def forward(
        self,
        points_xyz,
        features=None,
        indices=None,
        target_xyz=None,
    ):

        assert len(self.groupers) == 1

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)
        grouped_results = self.groupers[0](points_xyz, new_xyz, features)
        new_features = self.mlp(grouped_results)

        B, C, NP, NS = new_features.shape
        new_features = new_features.permute(0, 2, 1, 3).flatten(2).contiguous()

        new_features = self.transformer_layer(new_features).transpose(1, 2)
        new_features = self.after_mlp(new_features)

        return new_xyz, new_features, indices
