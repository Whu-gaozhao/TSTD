import torch
from mmcv.cnn import ConvModule, Scale, build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import auto_fp16
from mmcv.utils import Registry
from torch import nn as nn

from mmdet3d.ops import build_sa_module
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet

FU_MODULES = Registry('point_fusion_module')


def build_fu_module(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='AddFusionModule')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    module_type = cfg_.pop('type')
    if module_type not in FU_MODULES:
        raise KeyError(f'Unrecognized module type {module_type}')
    else:
        fu_module = FU_MODULES.get(module_type)

    module = fu_module(*args, **kwargs, **cfg_)

    return module


@FU_MODULES.register_module()
class AddFusionModule(nn.Module):

    def __init__(self, in_channels):
        super(AddFusionModule, self).__init__()

    def forward(self, x):
        c = x.shape[1] // 2
        xyz, rgb = x[:, :c], x[:, c:]

        return xyz + rgb


@FU_MODULES.register_module()
class ConcatFusionModule(nn.Module):

    def __init__(self, in_channels):
        super(ConcatFusionModule, self).__init__()

        self.conv = ConvModule(
            in_channels,
            in_channels // 2,
            1,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


@FU_MODULES.register_module()
class SelfAttentionFusion(MultiheadAttention):

    def __init__(self,
                 in_channels,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(SelfAttentionFusion, self).__init__(
            in_channels,
            in_channels // 64,  # num_heads
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.channels = in_channels // 2
        self.gamma = Scale(0)
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def forward(self, x):
        # x: b,c,n feature
        qkv = x.transpose(1, 2)
        out = self.attn(query=qkv, key=qkv, value=qkv, need_weights=False)[0]
        out = self.gamma(out) + qkv
        enhanced_xyz = self.norm(out).transpose(1, 2)
        return enhanced_xyz[:, :self.channels].contiguous()


@BACKBONES.register_module()
class MVE_PCT(BasePointNet):
    """

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(1024, 256, 64, 16),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 num_samples=(32, 32, 32, 32),
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256),
                              (256, 256, 512)),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='TwoStreamSAModule',
                     pool_mod='max',
                     use_xyz=False,
                     normalize_xyz=False),
                 fu_cfg=dict(type='AddFusionModule'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)

        self.SA_modules = nn.ModuleList()
        self.FU_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    first_layer=True if sa_index == 0 else False,
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            self.FU_modules.append(
                build_fu_module(in_channels=sa_out_channel * 2, cfg=fu_cfg))
            sa_in_channel = sa_out_channel

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        """
        xyz, features = self._split_point_feats(points)
        assert features is not None

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [points.transpose(1, 2)]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))
        sa_features[0] = features
        for i in range(self.num_sa):
            sa_features[i + 1] = self.FU_modules[i](sa_features[i + 1])
        ret = dict(
            sa_xyz=sa_xyz, sa_features=sa_features, sa_indices=sa_indices)
        return ret
