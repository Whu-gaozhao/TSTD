import torch
import torch.nn as nn
import pointops
import einops
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
from mmcv.runner import auto_fp16

from mmdet.models import BACKBONES
from mmcv.runner import BaseModule

from copy import deepcopy
from timm.models.layers import DropPath







def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().to(offset.device)

def  split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
        
class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat
    


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster
    
class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster
        
class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]

class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        coord = coord.cuda()
        offset = offset.cuda()
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points

class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset = points
        # B,C,N = feat.shape
        # feat = feat.reshape(B,N,C)
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])
    





@BACKBONES.register_module()
class  PointTransFormerV2(BaseModule):

    def __init__(self,
                 in_channels,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2,2,6),
                 enc_channels=(96,192,384),
                 enc_groups=(12,24,48),
                 enc_neighbours=(16,16,16),
                 grid_sizes=(0.1, 0.2, 0.4),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                #unpool_backend="map",
                              ):
        super(PointTransFormerV2,self).__init__()
        self.in_channels = in_channels
        # self.num_classes = num_classes
        self.num_stages = len(enc_depths)
         
        self.patch_embed = GVAPatchEmbed(
           in_channels = in_channels,
           embed_channels=patch_embed_channels,
           groups=patch_embed_groups,
           depth=patch_embed_depth,
           neighbours=patch_embed_neighbours,
           qkv_bias=attn_qkv_bias,
           pe_multiplier=pe_multiplier,
           pe_bias=pe_bias,
           attn_drop_rate=attn_drop_rate,
           enable_checkpoint=enable_checkpoint,
         )
        
        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)


        self.enc_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            self.enc_stages.append(enc)



    @auto_fp16(apply_to=('points', ))
    def forward(self,points):
        # points [B,N,C]
        B,N,C=points.shape
        xyz,feat = split_point_feats(points)
        # coord = torch.randn(B,N,C)
        xyz_c = xyz.shape[-1]

        #shift to PCR style [B*N,C]
        coord = xyz.reshape([-1,xyz_c])
        feat = feat.reshape([-1,C-xyz_c])
        offset = ((torch.arange(B,device=feat.device) + 1) * N).int()
        # offset = torch.tensor([B*N])
        # offset = offset.cuda()

        #a batch of pointcloud is [coord,feat,offset]
        points = [coord,feat,offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points,cluster = self.enc_stages[i](points)
            #record grid cluster of pooling
            skips[-1].append(cluster)
            #record points info of current stage
            skips.append([points])

#unpooling points info in the last enc stage
        points = skips.pop(-1)[0]
        ret = dict(points=points,skips=skips,B=B)
        
        return ret