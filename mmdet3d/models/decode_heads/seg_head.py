from mmcv.cnn.bricks import ConvModule
from mmcv.cnn import constant_init
from timm.models.layers import trunc_normal_

import torch
import math
from torch import nn as nn
from torch.nn import functional as F

from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead,Base3DDecodeHeadNew


class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context






class Class_Token_Seg3(nn.Module):
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, num_classes=20, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)


        self.cls_token = nn.Parameter(torch.zeros(1, num_classes, dim))
        self.prop_token = nn.Parameter(torch.zeros(1, num_classes, dim))
        
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.prop_token, std=.02)

    def forward(self, x):#, x1):
        # print(x.size)
        b, c, n = x.size()
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        prop_tokens = self.prop_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x[:, 0:self.num_classes]).unsqueeze(1).reshape(B, self.num_classes, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        k = k * self.scale
        attn = (k @ q.transpose(-2, -1)).squeeze(1).transpose(-2, -1) #k@q.transpose(-2,-1) 等价于 k.matmul(q.transpose(-2,-1))
        attn = attn[:, self.num_classes:]
        # x_cls = attn.permute(0, 2, 1).reshape(b, -1, h, w)
        x_cls = attn.permute(0, 2, 1).reshape(b, -1, n)
        return x_cls, prop_tokens

class TransformerClassToken3(nn.Module):

    def __init__(self, dim, num_heads=2, num_classes=20, depth=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_cfg=None, norm_cfg=None, sr_ratio=1, trans_with_mlp=True, att_type="SelfAttention"):
        super().__init__()
        self.trans_with_mlp = trans_with_mlp
        self.depth = depth
        print("TransformerOriginal initial num_heads:{}; depth:{}, self.trans_with_mlp:{}".format(num_heads, depth, self.trans_with_mlp))   
        self.num_classes = num_classes
        
        self.attn = SelfAttentionBlock(
            key_in_channels=dim,
            query_in_channels=dim,
            channels=dim,
            out_channels=dim,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            # conv_cfg=None,
            conv_cfg=dict(type='Conv1d'),
            # norm_cfg=norm_cfg,
            # act_cfg=act_cfg
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
             )

        self.cross_attn = SelfAttentionBlock(
            key_in_channels=dim,
            query_in_channels=dim,
            channels=dim,
            out_channels=dim,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            # conv_cfg=None,
            # norm_cfg=norm_cfg,
            # act_cfg=act_cfg
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            )
        
        #self.conv = nn.Conv2d(dim*3, dim, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(dim*2,dim, kernel_size=3, stride=1, padding=1)
        self.apply(self._init_weights) 
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
             
    def forward(self, x, cls_tokens, out_cls_mid):
        b, c, n = x.size()
        out_cls_mid = out_cls_mid.flatten(2).transpose(1, 2)  
        
        # x = x.unsqueeze(-1)

        #within images attention
        x1 = self.attn(x, x)

        #cross images attention
        out_cls_mid = out_cls_mid.softmax(dim=-1)
        cls = out_cls_mid @ cls_tokens #bxnxc
        
        cls = cls.permute(0, 2, 1).reshape(b, c, n)
        x2 = self.cross_attn(x, cls)

        x = x+x1+x2
        
        return x


@HEADS.register_module()
class SegDeformerHead(Base3DDecodeHeadNew):
    def __init__(self,
                in_channels=1024,
                mid_channels=512, 
                # interpolate_mode='bilinear',
                num_heads=4,
                m=0.9,
                trans_with_mlp=True,
                trans_depth=1,
                align_corners=False,
                att_type="XCA",
                **kwargs):
        # super().__init__(input_transform='multiple_select',**kwargs)
        super(SegDeformerHead, self).__init__(**kwargs)
        # self.interpolate_mode = interpolate_mode
        # num_inputs = len(self.in_channels)
        self.num_heads = num_heads
        self.align_corners=align_corners

        # self.mid_channels = mid_channels


        self.pre_seg_conv = ConvModule(
            mid_channels,
            self.channels,
            1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv = ConvModule(
            3 * in_channels,
            mid_channels,
            1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


        self.class_token = Class_Token_Seg3(dim=self.channels, 
                                            num_heads=1,
                                            num_classes=self.num_classes)
        self.trans = TransformerClassToken3(dim=self.channels, 
                                            depth=trans_depth, 
                                            num_heads=self.num_heads, 
                                            trans_with_mlp=trans_with_mlp, 
                                            att_type=att_type,
                                             norm_cfg=self.norm_cfg,
                                             act_cfg=self.act_cfg)





    def forward(self,x,gt_semantic_seg=None):
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
        out_cls_mid,cls_tokens = self.class_token(output)
        out_new = self.trans(output,cls_tokens,out_cls_mid)
        output = self.cls_seg(out_new)

        return output,out_cls_mid