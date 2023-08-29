import torch
import torch.nn as nn
from mmcv.runner import auto_fp16

from mmdet3d.models.utils import TransformerBlock
from mmdet3d.ops import PointFPModule
from mmdet3d.ops import build_sa_module
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet



@BACKBONES.register_module()
class PointTransformer(BasePointNet):

    def __init__(
        self,
        in_channels,
        embed_dim=32,
        # embed_dim=16,
        num_point=1024,
        # num_point=4096,
        num_stage=4,
        num_neighbor=16,
        # neck_channel=512,
        transformer_channel=512,
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True),
    ):
        super().__init__()
        self.num_stage = num_stage
        # self.embed_mlp = nn.Sequential(
        #     nn.Linear(in_channels, in_channels-3), nn.ReLU(),
        #     nn.Linear(in_channels-3, in_channels-3))
        # self.embed_transformer = TransformerBlock(in_channels-3,
        #                                           transformer_channel,
        #                                           num_neighbor)

        self.transition_downs = nn.ModuleList()
        # self.transformers = nn.ModuleList()


        for i in range(num_stage):
            channel = embed_dim * 2**(i + 1)
            in_channel = channel // 2 if i > 0 else 0
            self.transition_downs.append(
                build_sa_module(
                    num_point=num_point // 8**(i + 1),
                    radius=None,
                    num_sample=num_neighbor,
                    mlp_channels=[
                        in_channel, channel // 2, channel // 2, channel
                    ],
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            # self.transformers.append(
            #     TransformerBlock(channel, transformer_channel, num_neighbor))

        # self.neck_mlp = nn.Sequential(
        #     nn.Linear(embed_dim * 2**num_stage, neck_channel), nn.ReLU(),
        #     nn.Linear(neck_channel, neck_channel), nn.ReLU(),
        #     nn.Linear(neck_channel, embed_dim * 2**num_stage))
        # self.neck_transformer = TransformerBlock(embed_dim * 2**num_stage,
        #                                          transformer_channel,
        #                                          num_neighbor)

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):  # points: B,N,C
        xyz, feat = self._split_point_feats(points)
        # feat = self.embed_mlp(points).transpose(1, 2).contiguous()
        # feat = self.embed_transformer(xyz, feat)[0]
        # feat = None
        xyz_list = [xyz]
        feat_list = [None] # early middle feat  late None
        for i in range(self.num_stage):
            cur_xyz, cur_feat, _ = self.transition_downs[i](xyz_list[i], feat_list[i])
            # cur_feat = self.transformers[i](cur_xyz, cur_feat)[0]
            xyz_list.append(cur_xyz)
            feat_list.append(cur_feat)
        # feat_list[0]=feat  # xyz early None middle late feat
        # feat = feat.transpose(1, 2)
        # feat = self.neck_mlp(feat).transpose(1, 2).contiguous()
        # feat_list[-1] = self.neck_transformer(xyz, feat)[0]

        ret = dict(xyz=xyz_list, feats=feat_list)

        
        return ret
