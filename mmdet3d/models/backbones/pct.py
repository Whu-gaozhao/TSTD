import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, auto_fp16
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.ops import PointFPModule, build_sa_module
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet
from .mve_pct import FU_MODULES


@FU_MODULES.register_module()
class SA_Layer(nn.Module):

    def __init__(self,
                 in_channels,
                 normal_type='SL',
                 offset=True,
                 slice=False):
        super(SA_Layer, self).__init__()
        channels = in_channels
        assert normal_type in ['SL', 'SS']
        self.channels = channels
        self.normal_type = normal_type
        self.offset = offset
        self.slice = slice

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.LBR = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        if self.normal_type == 'SS':
            attention = F.softmax(energy, dim=-1)
            attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        else:
            energy = (self.channels**-.5) * energy
            attention = F.softmax(energy, dim=-2)
        x_r = torch.bmm(x_v, attention)  # b, c, n

        if self.offset:
            x_r = x - x_r
        x = x + self.LBR(x_r)
        if self.slice:
            c = x.shape[1] // 2
            x = x[:, :c].contiguous()
        return x


@BACKBONES.register_module()
class Point_Transformer(BaseModule):
    """Point_Transformer single scale.

    Args:

    """

    def __init__(self,
                 in_channels,
                 out_channels=1024,
                 channels=128,
                 num_stages=4,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(Point_Transformer, self).__init__(init_cfg=init_cfg)

        self.num_stages = num_stages

        self.point_embedding = ConvModule(
            in_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.sa = ModuleList()
        for _ in range(self.num_stages):
            self.sa.append(SA_Layer(channels))

        self.conv_fuse = ConvModule(
            channels * num_stages,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

    @auto_fp16()
    def forward(self, points):
        # b, n, c --> b, c, n
        x = points.permute(0, 2, 1)
        x = self.point_embedding(x)
        x = self.conv(x)

        features = []
        for i in range(self.num_stages):
            x = self.sa[i](x)
            features.append(x)
        x = torch.cat(features, dim=1)
        out = self.conv_fuse(x)
        return out


@BACKBONES.register_module()
class PCT(BasePointNet):
    """Multi-scale Point_Transformer based on Pointnet++ and offset attention.

    Args:

    """

    def __init__(self,
                 in_channels,
                 num_points=(1024, 256, 64, 16),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 num_samples=(32, 32, 32, 32),
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256),
                              (256, 256, 512)),
                 attention_index=[0, 1, 2, 3],
                 attention_cfg=dict(normal_type='SL', offset=True),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.attention_index = attention_index

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(attention_index)

        self.SA_modules = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
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
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            self.attention_modules.append(
                SA_Layer(sa_out_channel, **attention_cfg))
            sa_in_channel = sa_out_channel

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        for i in range(self.num_sa):
            if i in self.attention_index:
                sa_features[i + 1] = self.attention_modules[i](
                    sa_features[i + 1])

        ret = dict(
            sa_xyz=sa_xyz, sa_features=sa_features, sa_indices=sa_indices)
        return ret
