import torch
from mmcv.cnn import ConvModule
from torch import nn as nn

from .builder import SA_MODULES
from .point_sa_module import BasePointSAModule


@SA_MODULES.register_module()
class TwoStreamSAModuleMSG(BasePointSAModule):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 first_layer=False,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=False,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(TwoStreamSAModuleMSG, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        self.use_xyz = use_xyz
        self.xyz_mlps = nn.ModuleList()
        self.rgb_mlps = nn.ModuleList()
        for i in range(len(self.mlp_channels)):
            xyz_mlp_channel = self.mlp_channels[i]
            rgb_mlp_channel = self.mlp_channels[i].copy()

            if first_layer:
                xyz_mlp_channel[0] = 3
            if self.use_xyz:
                xyz_mlp_channel[0] += 3

            xyz_mlp = nn.Sequential()
            for i in range(len(xyz_mlp_channel) - 1):
                xyz_mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        xyz_mlp_channel[i],
                        xyz_mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.xyz_mlps.append(xyz_mlp)

            rgb_mlp = nn.Sequential()
            for i in range(len(rgb_mlp_channel) - 1):
                rgb_mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        rgb_mlp_channel[i],
                        rgb_mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.rgb_mlps.append(rgb_mlp)

    @staticmethod
    def _split_xyz_rgb_features(features, use_xyz=False):
        """

        Args:
            features (Tensor): (B, C, num_point, nsample) Grouped features

        """
        c = features.shape[1]
        if use_xyz:
            split_channel = (c + 3) // 2 if c % 2 != 0 else 6
        else:
            split_channel = c // 2 if c % 2 == 0 else 3
        xyz = features[:, :split_channel].contiguous()
        rgb = features[:, split_channel:].contiguous()
        return xyz, rgb

    def forward(self, points_xyz, features):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_xyz_features_list = []
        new_rgb_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, None,
                                               None)

        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz,
                                               features.contiguous())
            xyz_grouped_results, rgb_grouped_results = self._split_xyz_rgb_features(
                grouped_results, self.use_xyz)

            # (B, mlp[-1], num_point, nsample)
            new_xyz_features = self.xyz_mlps[i](xyz_grouped_results)
            new_rgb_features = self.rgb_mlps[i](rgb_grouped_results)

            # (B, mlp[-1], num_point)
            new_xyz_features = self._pool_features(new_xyz_features)
            new_rgb_features = self._pool_features(new_rgb_features)
            new_xyz_features_list.append(new_xyz_features)
            new_rgb_features_list.append(new_rgb_features)

        new_features = torch.cat(
            new_xyz_features_list + new_rgb_features_list, dim=1)
        return new_xyz, new_features, indices


@SA_MODULES.register_module()
class TwoStreamSAModule(TwoStreamSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 mlp_channels,
                 first_layer=False,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 normalize_xyz=False):
        super(TwoStreamSAModule, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            first_layer=first_layer,
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)
