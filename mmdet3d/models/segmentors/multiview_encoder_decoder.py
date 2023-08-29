import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmseg.core import add_prefix
from mmseg.models import SEGMENTORS
from .. import builder
from .encoder_decoder import EncoderDecoder3D


@SEGMENTORS.register_module()
class MultivewEncoderDecoder(EncoderDecoder3D):
    r"""

    """

    def __init__(self, img_backbone, fusion_layer, img_neck=None, **kwargs):
        super(MultivewEncoderDecoder, self).__init__(**kwargs)

        # image branch
        self.img_backbone = builder.build_seg_backbone(img_backbone)
        self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    def extract_img_feat(self, img):
        """Directly extract features from the img backbone+neck."""
        # point fusion layer only supports single view input now
        img = img.squeeze(dim=1)
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def extract_fused_feat(self, img, pts, pts_feature, img_metas):
        img_feature = self.extract_img_feat(img)
        fused_feature = self.fusion_layer(img_feature, pts, pts_feature,
                                          img_metas)
        return fused_feature

    def encode_decode(self, points, img, img_metas):
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, N].
        """
        pts_feature = self.extract_feat(points)
        x = self.extract_fused_feat(img, points, pts_feature, img_metas)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def forward_dummy(self, points, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(points, img, None)

        return seg_logit

    def forward_train(self, points, img, img_metas, pts_semantic_mask):
        """Forward function for training.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, C].
            img (torch.Tensor): Multi-batch and Multi-view images 
                of shape  [B, N, C, H, W].
            img_metas (list): Image metas.
            pts_semantic_mask (list[torch.Tensor]): List of point-wise semantic
                labels of shape [N].

        Returns:
            dict[str, Tensor]: Losses.
        """
        points_cat = torch.stack(points)
        pts_semantic_mask_cat = torch.stack(pts_semantic_mask)

        # extract features using backbone
        pts_feature = self.extract_feat(points_cat)
        x = self.extract_fused_feat(img, points_cat, pts_feature, img_metas)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      pts_semantic_mask_cat)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, pts_semantic_mask_cat)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    def slide_inference(self, point, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        Args:
            point (torch.Tensor): Input points of shape [N, 3+C].
            img_meta (dict): Meta information of input sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map of shape [num_classes, N].
        """
        num_points = self.test_cfg.num_points
        block_size = self.test_cfg.block_size
        sample_rate = self.test_cfg.sample_rate
        use_normalized_coord = self.test_cfg.use_normalized_coord
        batch_size = self.test_cfg.batch_size * num_points

        # patch_points is of shape [K*N, 3+C], patch_idxs is of shape [K*N]
        patch_points, patch_idxs = self._sliding_patch_generation(
            point, num_points, block_size, sample_rate, use_normalized_coord)
        feats_dim = patch_points.shape[1]
        seg_logits = []  # save patch predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points, img, img_meta)
            batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        expand_patch_idxs = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        preds = point.new_zeros((point.shape[0], self.num_classes)).\
            scatter_add_(dim=0, index=expand_patch_idxs, src=seg_logits)
        count_mat = torch.bincount(patch_idxs)
        preds = preds / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        return preds.transpose(0, 1)  # to [num_classes, K*N]

    def whole_inference(self, points, img, img_metas, rescale):
        """Inference with full scene (one forward pass without sliding)."""
        seg_logit = self.encode_decode(points, img, img_metas)
        # TODO: if rescale and voxelization segmentor
        return seg_logit

    def inference(self, points, img, img_metas, rescale):
        """Inference with slide/whole style.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = torch.stack([
                self.slide_inference(point, img, img_meta, rescale)
                for point, img_meta in zip(points, img_metas)
            ], 0)
        else:
            seg_logit = self.whole_inference(points, img, img_metas, rescale)
        output = F.softmax(seg_logit, dim=1)
        return output

    def simple_test(self, points, img_metas, img, rescale=True):
        """Simple test with single scene.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        seg_pred = []
        # print("\n")
        # print(len(points), points[0].shape)
        # print(len(img_metas), img_metas[0])
        # print(len(img), img[0].shape)
        for point, _img, img_meta in zip(points, img, img_metas):
            seg_prob = self.inference(
                point.unsqueeze(0), _img, [img_meta], rescale)[0]
            seg_map = seg_prob.argmax(0)  # [N]
            # to cpu tensor for consistency with det3d
            seg_map = seg_map.cpu()
            seg_pred.append(seg_map)
        # warp in dict
        seg_pred = [dict(semantic_mask=seg_map) for seg_map in seg_pred]
        return seg_pred