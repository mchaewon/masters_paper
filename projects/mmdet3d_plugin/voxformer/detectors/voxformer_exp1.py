# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import time
import copy
import torch
import torch.nn as nn
import numpy as np
import mmdet3d
# from tkinter.messagebox import NO
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.bricks import run_time

@DETECTORS.register_module()
class VoxFormerExp1(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):

        super(VoxFormerExp1,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        # Exp1: 첫 conv 3ch -> 6ch (RGB pretrained + Event zero-init)
        self._expand_first_conv_to_6ch()

        # ── Exp2: EventEncoder + Fusion ──────────────────────
        event_feat_dim = 128   # FPN output과 동일 (_dim_)
        self.event_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, event_feat_dim, 1),
        )
        # RGB feat + Event feat → fused feat
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(event_feat_dim * 2, event_feat_dim, 1),
            nn.BatchNorm2d(event_feat_dim),
            nn.ReLU(),
        )
        print("[Exp2] EventEncoder + FusionConv 초기화 완료")
        # ── Exp2 End ──────────────────────────────────────────



    def extract_event_feat(self, event_voxel):
        """
        Event voxel grid (B, T, H, W) → 2D feature map (B, C, H', W')
        T = num_bins (5), C = 128 (FPN output dim과 동일)
        """
        B, N, T, H, W = event_voxel.shape
        # 현재 프레임만 사용 (마지막 프레임 = 최신)
        ev = event_voxel[:, -1]   # (B, T, H, W)
        ev_feat = self.event_encoder(ev)  # (B, C, H', W')
        return ev_feat


    def _expand_first_conv_to_6ch(self):
        first_conv = self.img_backbone.conv1
        if first_conv.in_channels == 6:
            return
        old_w = first_conv.weight.data.clone()
        new_conv = nn.Conv2d(6, first_conv.out_channels,
                             first_conv.kernel_size, first_conv.stride,
                             first_conv.padding, bias=False)
        new_conv.weight.data[:, :3] = old_w
        new_conv.weight.data[:, 3:] = 0.0
        self.img_backbone.conv1 = new_conv
        print("[Exp1] First conv: 3ch -> 6ch")

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        losses = self.pts_bbox_head.training_step(outs, target, img_metas)
        return losses

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      target=None):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]

        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img)

        # ── Exp2: Event feature 추출 및 fusion ───────────────
        # ─────────────────────────────────────────────────────

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     img_metas=None,
                     img=None,
                     target=None,
                      **kwargs):
        """Forward testing function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]

        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img)

        # ── Exp2: Event feature 추출 및 fusion ───────────────
        # ─────────────────────────────────────────────────────

        outs = self.pts_bbox_head(img_feats, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        if isinstance(completion_results, dict):
            completion_results = [completion_results]

        return completion_results
