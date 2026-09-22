"""
VoxFormerExp3Sep: F_ego/F_obj 분리 3D Cross-Attention Detector
===============================================================
Exp3 3D와의 차이:
  Exp3 3D: ev_proj(concat(F_ego, F_obj)) → ev_feats 1개 → Head에 전달
  Exp3 Sep: F_ego, F_obj 분리 → Head에 각각 전달
            → Head에서 3번의 독립적 3D cross-attention 수행
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from .voxformer_exp3_diff import (
    EgoEventEncoderGN,
    MotionEventEncoderGN,
    load_event_voxel_batch,
    load_depth_batch,
    load_moving_mask_batch,
)


@DETECTORS.register_module()
class VoxFormerExp3Sep(MVXTwoStageDetector):
    """
    Separated 3D event cross-attention.
    pts_bbox_head: VoxFormerHeadEventV2 사용 필수.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None, pts_bbox_head=None,
                 img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 event_channels=128, num_event_bins=5, num_groups=8,
                 lambda_ego=0.1, lambda_obj=0.1,
                 img_H=370, img_W=1220, ev_H=352, ev_W=1216,
                 ):
        super().__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck,
            pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )

        self.ego_encoder = EgoEventEncoderGN(event_channels, num_groups)
        self.obj_encoder = MotionEventEncoderGN(num_event_bins, event_channels, num_groups)

        # F_ego, F_obj 각각 별도 projection (분리 유지!)
        self.ego_proj = nn.Sequential(
            nn.Conv2d(event_channels, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels),
            nn.ReLU(inplace=True),
        )
        self.obj_proj = nn.Sequential(
            nn.Conv2d(event_channels, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels),
            nn.ReLU(inplace=True),
        )

        self.lambda_ego = lambda_ego
        self.lambda_obj = lambda_obj
        self.T   = num_event_bins
        self.img_H, self.img_W = img_H, img_W
        self.ev_H,  self.ev_W  = ev_H,  ev_W
        print("[VoxFormerExp3Sep] F_ego/F_obj 분리 3D cross-attention")

    def _get_event_feats(self, img_metas, fpn_shape, device):
        """
        F_ego, F_obj를 분리해서 FPN 해상도로 반환.
        합치지 않음!
        """
        ev = load_event_voxel_batch(
            img_metas, self.T, self.ev_H, self.ev_W,
            self.img_H, self.img_W, device
        )
        F_ego = self.ego_encoder(ev)

        obj_out = self.obj_encoder(ev)
        F_obj = obj_out[0] if isinstance(obj_out, (tuple, list)) else obj_out

        # FPN 해상도로 align
        B, N, C, Hf, Wf = fpn_shape
        ego_feats = self.ego_proj(
            F.interpolate(F_ego, (Hf, Wf), mode='bilinear', align_corners=False)
        )  # (B, C, Hf, Wf)
        obj_feats = self.obj_proj(
            F.interpolate(F_obj, (Hf, Wf), mode='bilinear', align_corners=False)
        )  # (B, C, Hf, Wf)

        return ev, F_ego, F_obj, ego_feats, obj_feats

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        losses = {}
        depth_gt   = load_depth_batch(img_metas, self.img_H, self.img_W, device=device)
        depth_pred = self.ego_encoder.predict_depth(F_ego)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(depth_pred, depth_gt.shape[2:],
                                        mode='bilinear', align_corners=False)
        vd = (depth_gt > 0).float()
        losses['loss_ego_depth'] = (
            F.l1_loss(depth_pred*vd, depth_gt*vd, reduction='sum')
            / (vd.sum()+1e-6)
        ) * self.lambda_ego if vd.sum() > 0 else depth_pred.sum()*0.

        moving_gt = load_moving_mask_batch(img_metas, self.img_H, self.img_W, device=device)
        dyn_pred  = self.obj_encoder.predict_dynamic(F_obj)
        if dyn_pred.shape != moving_gt.shape:
            dyn_pred = F.interpolate(dyn_pred, moving_gt.shape[2:],
                                      mode='bilinear', align_corners=False)
        vm = (moving_gt >= 0).float()
        losses['loss_obj_dyn'] = (
            F.binary_cross_entropy_with_logits(
                dyn_pred*vm, (moving_gt>=0.5).float()*vm, reduction='sum'
            ) / (vm.sum()+1e-6)
        ) * self.lambda_obj if vm.sum() > 0 else dyn_pred.sum()*0.
        return losses

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B,N,C,H,W = img.size(); img = img.reshape(B*N,C,H,W)
            feats = self.img_backbone(img)
            if isinstance(feats, dict): feats = list(feats.values())
        else:
            return None
        if self.with_img_neck: feats = self.img_neck(feats)
        out = []
        for f in feats:
            BN,C,H,W = f.size()
            out.append(f.view(int(B/len_queue),len_queue,int(BN/B),C,H,W)
                       if len_queue else f.view(B,int(BN/B),C,H,W))
        return out

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue)

    def forward(self, return_loss=True, **kw):
        return self.forward_train(**kw) if return_loss else self.forward_test(**kw)

    @auto_fp16(apply_to=('img','points'))
    def forward_train(self, img_metas=None, img=None, target=None):
        lq    = img.size(1)
        metas = [each[lq-1] for each in img_metas]
        img   = img[:,-1,...]
        dev   = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj, ego_feats, obj_feats = self._get_event_feats(
            metas, img_feats[0].shape, dev
        )

        losses = {}
        # VoxFormerHeadEventV2: F_ego, F_obj 분리해서 전달
        outs = self.pts_bbox_head(
            img_feats, metas, target,
            ego_feats=ego_feats,  # ← 분리 전달
            obj_feats=obj_feats,  # ← 분리 전달
        )
        losses.update(self.pts_bbox_head.training_step(outs, target, metas))
        losses.update(self.compute_aux_losses(F_ego, F_obj, metas, dev))
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kw):
        lq    = img.size(1)
        metas = [each[lq-1] for each in img_metas]
        img   = img[:,-1,...]
        dev   = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj, ego_feats, obj_feats = self._get_event_feats(
            metas, img_feats[0].shape, dev
        )
        outs = self.pts_bbox_head(
            img_feats, metas, target,
            ego_feats=ego_feats,
            obj_feats=obj_feats,
        )
        res = self.pts_bbox_head.validation_step(outs, target, metas)
        return [res] if isinstance(res, dict) else res