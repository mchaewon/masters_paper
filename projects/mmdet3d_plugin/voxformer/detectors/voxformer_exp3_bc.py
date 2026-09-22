"""
VoxFormerExp3BC: Dual Event Encoder + Method B + C
===================================================
Exp3GN 구조에 Method B+C를 추가:
  Method C: F_obj → event query → proposal augmentation
  Method B: event cross-attn stream → gated 3D fusion

VoxFormerHead 대신 VoxFormerHeadBC를 사용.
나머지(EgoEncoder, ObjEncoder, FusionModule)는 Exp3GN과 동일.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

# Exp3GN의 encoder/fusion 재사용
from .voxformer_exp3_gn import (
    EgoEventEncoderGN,
    ObjEventEncoderGN,
    load_event_voxel_batch,
    load_depth_batch,
    load_moving_mask_batch,
)


@DETECTORS.register_module()
class VoxFormerExp3BC(MVXTwoStageDetector):
    """
    VoxFormerExp3GN + Method B (dual-stream lifting)
                    + Method C (event query augmentation)

    변경사항:
      - pts_bbox_head: VoxFormerHeadBC 사용
      - forward에서 event_feat_2d, F_obj, depth_map을
        pts_bbox_head에 전달
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None,
                 pts_bbox_head=None,   # VoxFormerHeadBC
                 img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 # Exp3 파라미터
                 event_channels=128,
                 num_event_bins=5,
                 num_groups=8,
                 lambda_ego=0.1,
                 lambda_obj=0.1,
                 img_H=370, img_W=1220,
                 ev_H=352, ev_W=1216,
                 ):
        super().__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck,
            pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )

        # Dual Event Encoder (Exp3GN과 동일)
        self.ego_encoder = EgoEventEncoderGN(
            out_channels=event_channels, num_groups=num_groups
        )
        self.obj_encoder = ObjEventEncoderGN(
            num_bins=num_event_bins,
            out_channels=event_channels,
            num_groups=num_groups
        )

        # 2D fusion: event_feat_2d = proj(concat(F_ego, F_obj))
        # VoxFormerHead에 넘길 event 2D feature 생성
        self.event_2d_proj = nn.Sequential(
            nn.Conv2d(event_channels * 2, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels),
            nn.ReLU(inplace=True)
        )

        self.lambda_ego = lambda_ego
        self.lambda_obj = lambda_obj
        self.num_event_bins = num_event_bins
        self.img_H, self.img_W = img_H, img_W
        self.ev_H,  self.ev_W  = ev_H,  ev_W

        print("[VoxFormerExp3BC] Method B+C enabled")

    def extract_event_feats(self, img_metas, device):
        """Event voxel → F_ego, F_obj, event_feat_2d"""
        ev = load_event_voxel_batch(
            img_metas,
            num_bins=self.num_event_bins,
            ev_H=self.ev_H, ev_W=self.ev_W,
            img_H=self.img_H, img_W=self.img_W,
            device=device
        )  # (B, bins, H, W)

        F_ego = self.ego_encoder(ev)   # (B, C, H/4, W/4)
        F_obj = self.obj_encoder(ev)   # (B, C, H/4, W/4)

        # FPN 해상도로 맞춰서 concat → event_feat_2d
        # (VoxFormerHeadBC의 EventCrossAttn에 전달)
        target_size = img_metas[0].get(
            'img_shape', [(self.img_H, self.img_W)]
        )[0]
        # FPN output 해상도 추정 (H/16)
        # 메모리 절감: EventCrossAttn의 sequence length 축소
        feat_h = self.img_H // 16
        feat_w = self.img_W // 16

        ego_r = F.interpolate(F_ego, (feat_h, feat_w),
                               mode='bilinear', align_corners=False)
        obj_r = F.interpolate(F_obj, (feat_h, feat_w),
                               mode='bilinear', align_corners=False)
        event_feat_2d = self.event_2d_proj(
            torch.cat([ego_r, obj_r], dim=1)
        )  # (B, C, feat_h, feat_w)

        return ev, F_ego, F_obj, event_feat_2d

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        """Auxiliary supervision (Exp3GN과 동일)"""
        losses = {}

        # Ego: depth L1
        depth_gt   = load_depth_batch(img_metas, self.img_H, self.img_W,
                                       device=device)
        depth_pred = self.ego_encoder.predict_depth(F_ego)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(depth_pred, size=depth_gt.shape[2:],
                                        mode='bilinear', align_corners=False)
        valid_d = (depth_gt > 0).float()
        if valid_d.sum() > 0:
            losses['loss_ego_depth'] = (
                F.l1_loss(depth_pred * valid_d, depth_gt * valid_d,
                          reduction='sum') / (valid_d.sum() + 1e-6)
            ) * self.lambda_ego
        else:
            losses['loss_ego_depth'] = depth_pred.sum() * 0.

        # Obj: moving mask BCE
        moving_gt  = load_moving_mask_batch(img_metas, self.img_H, self.img_W,
                                             device=device)
        dyn_pred   = self.obj_encoder.predict_dynamic(F_obj)
        if dyn_pred.shape != moving_gt.shape:
            dyn_pred = F.interpolate(dyn_pred, size=moving_gt.shape[2:],
                                      mode='bilinear', align_corners=False)
        valid_m = (moving_gt >= 0).float()
        if valid_m.sum() > 0:
            gt_bin = (moving_gt >= 0.5).float()
            losses['loss_obj_dyn'] = (
                F.binary_cross_entropy_with_logits(
                    dyn_pred * valid_m, gt_bin * valid_m, reduction='sum'
                ) / (valid_m.sum() + 1e-6)
            ) * self.lambda_obj
        else:
            losses['loss_obj_dyn'] = dyn_pred.sum() * 0.

        return losses

    def extract_img_feat(self, img, img_metas, len_queue=None):
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
        for feat in img_feats:
            BN, C, H, W = feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    feat.view(int(B/len_queue), len_queue, int(BN/B), C, H, W)
                )
            else:
                img_feats_reshaped.append(
                    feat.view(B, int(BN/B), C, H, W)
                )
        return img_feats_reshaped

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

    def forward(self, return_loss=True, **kwargs):
        return self.forward_train(**kwargs) if return_loss \
               else self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img_metas=None, img=None, target=None):
        len_queue = img.size(1)
        img_metas = [each[len_queue - 1] for each in img_metas]
        img       = img[:, -1, ...]
        device    = img.device

        # 1. RGB features
        img_feats = self.extract_feat(img=img)

        # 2. Event features
        _, F_ego, F_obj, event_feat_2d = self.extract_event_feats(
            img_metas, device
        )

        # 3. Depth map for Method C
        depth_map = load_depth_batch(
            img_metas, self.img_H, self.img_W, device=device
        )

        # 4. VoxFormerHeadBC forward
        #    Method C: F_obj + depth_map → proposal augmentation (내부)
        #    Method B: event_feat_2d → event cross-attn (내부)
        losses = dict()
        outs = self.pts_bbox_head(
            img_feats, img_metas, target,
            event_feat_2d=event_feat_2d,  # Method B
            F_obj=F_obj,                   # Method C
            depth_map=depth_map,           # Method C
        )
        losses_pts = self.pts_bbox_head.training_step(
            outs, target, img_metas
        )
        losses.update(losses_pts)

        # 5. Auxiliary losses
        losses.update(
            self.compute_aux_losses(F_ego, F_obj, img_metas, device)
        )
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kwargs):
        len_queue = img.size(1)
        img_metas = [each[len_queue - 1] for each in img_metas]
        img       = img[:, -1, ...]
        device    = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj, event_feat_2d = self.extract_event_feats(
            img_metas, device
        )

        # 추론 시 depth_map (Method C는 학습 시에만 동작)
        depth_map = load_depth_batch(
            img_metas, self.img_H, self.img_W, device=device
        )

        outs = self.pts_bbox_head(
            img_feats, img_metas, target,
            event_feat_2d=event_feat_2d,
            F_obj=F_obj,
            depth_map=depth_map,
        )
        results = self.pts_bbox_head.validation_step(outs, target, img_metas)
        if isinstance(results, dict):
            results = [results]
        return results