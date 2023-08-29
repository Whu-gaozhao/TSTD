import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule

from mmdet3d.models.utils import TransformerBlock
from mmdet3d.ops import PointFPModule
from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead


class SwapAxes(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2).contiguous()


class TransitionUp(nn.Module):

    def __init__(self,
                 low_channel,
                 high_channel,
                 out_channel,
                 fp_norm_cfg=dict(type='BN2d')):

        super().__init__()
        self.fc1 = nn.Sequential(
            SwapAxes(),
            nn.Linear(low_channel, out_channel),
            SwapAxes(),
            nn.BatchNorm1d(out_channel),  # TODO
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            SwapAxes(),
            nn.Linear(high_channel, out_channel),
            SwapAxes(),
            nn.BatchNorm1d(out_channel),  # TODO
            nn.ReLU(),
        )
        self.fp = PointFPModule(mlp_channels=[], norm_cfg=fp_norm_cfg)

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats1 = self.fp(xyz2, xyz1, None, feats1)
        if points2 is None:
            return feats1
        else:
            feats2 = self.fc2(points2)
            return feats1 + feats2


@HEADS.register_module()
class PointTransformerHead(Base3DDecodeHead):

    def __init__(self,
                 embed_dim=32,
                 num_stage=4,
                 num_neighbor=16,
                 transformer_channel=512,
                 **kwargs):
        super(PointTransformerHead, self).__init__(**kwargs)
        self.num_stage = num_stage
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(num_stage)):
            channel = embed_dim * 2**i
            high_channel = channel if i > 0 else 20 
            self.transition_ups.append(
                TransitionUp(channel * 2, high_channel, channel))
            self.transformers.append(
                TransformerBlock(channel, transformer_channel, num_neighbor))

        self.mlp = nn.Sequential(
            SwapAxes(),
            nn.Linear(embed_dim, self.channels),
            nn.ReLU(),
            nn.Linear(self.channels, self.channels),
            nn.ReLU(),
            SwapAxes(),
        )

    def forward(self, feat_dict):
        xyz, feats = feat_dict['xyz'], feat_dict['feats']
        feat = feats[-1]
        for i in range(self.num_stage):
            feat = self.transition_ups[i](xyz[-i - 1], feat, xyz[-i - 2],
                                          feats[-i - 2])
            if i!= self.num_stage-1:
                feat = self.transformers[i](xyz[-i - 2], feat)[0]
   
        output = self.mlp(feat)
        output = self.cls_seg(output)
        return output
