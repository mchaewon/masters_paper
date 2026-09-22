"""
VoxFormerExp3Hybrid: 2D Fusion + 3D Geometry-Aware Event Sampling
=================================================================
Exp3 Diff + Exp3 3D 통합:
  [2D] MotionGatedFusion → img_feats 보강 (truck/car↑)
  [3D] VoxFormerHeadEvent 3D cross-attention (person/bicycle↑)

기대 효과:
  truck:  Exp3 Diff 수준 유지 (~15%)
  person: Exp3 3D 수준 유지  (~1.2%)
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


class MotionGatedFusion(nn.Module):
    """2D motion-conditioned fusion (Exp3 Diff 방식)"""
    def __init__(self, rgb_ch=128, ev_ch=128, num_groups=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(rgb_ch + ev_ch * 2, rgb_ch, 1, bias=False),
            nn.GroupNorm(num_groups, rgb_ch), nn.ReLU(inplace=True),
        )

    def forward(self, img_feats, F_ego, F_obj, motion_mag):
        out = []
        for feat in img_feats:
            B, N, C, Hf, Wf = feat.shape
            ego_r = F.interpolate(F_ego, (Hf, Wf), mode='bilinear', align_corners=False)
            obj_r = F.interpolate(F_obj, (Hf, Wf), mode='bilinear', align_corners=False)
            mag_r = F.interpolate(motion_mag, (Hf, Wf), mode='bilinear', align_corners=False)
            g     = self.gate(mag_r)
            ego_w = ((1-g)*ego_r).unsqueeze(1).expand(B,N,C,Hf,Wf)
            obj_w = (   g *obj_r).unsqueeze(1).expand(B,N,C,Hf,Wf)
            cat   = torch.cat([feat, ego_w, obj_w], 2).view(B*N, 3*C, Hf, Wf)
            out.append(self.proj(cat).view(B,N,C,Hf,Wf))
        return out


@DETECTORS.register_module()
class VoxFormerExp3Hybrid(MVXTwoStageDetector):
    """
    Hybrid: 2D fusion (Diff) + 3D event cross-attention.
    pts_bbox_head: VoxFormerHeadEvent 사용 필수.
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

        self.ego_encoder  = EgoEventEncoderGN(event_channels, num_groups)
        self.obj_encoder  = MotionEventEncoderGN(num_event_bins, event_channels, num_groups)

        # 2D fusion (Exp3 Diff 방식 - truck/car 향상)
        self.fusion_2d = MotionGatedFusion(128, event_channels, num_groups)

        # 3D cross-attention용 ev_feats projection
        self.ev_proj = nn.Sequential(
            nn.Conv2d(event_channels * 2, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels), nn.ReLU(inplace=True),
        )

        self.lambda_ego = lambda_ego
        self.lambda_obj = lambda_obj
        self.T   = num_event_bins
        self.img_H, self.img_W = img_H, img_W
        self.ev_H,  self.ev_W  = ev_H,  ev_W
        print("[VoxFormerExp3Hybrid] 2D fusion + 3D geometry-aware sampling")

    def _extract_event(self, img_metas, img_feats, device):
        ev = load_event_voxel_batch(
            img_metas, self.T, self.ev_H, self.ev_W,
            self.img_H, self.img_W, device
        )
        F_ego = self.ego_encoder(ev)
        obj_out    = self.obj_encoder(ev)
        # 반환값 타입에 무관하게 안전하게 처리
        if isinstance(obj_out, (tuple, list)):
            F_obj      = obj_out[0]
            motion_mag = obj_out[1] if len(obj_out) > 1 else None
        else:
            F_obj      = obj_out
            motion_mag = None

        # motion_mag 없으면 DIFF magnitude를 직접 계산
        if motion_mag is None:
            diffs = ev[:, 1:] - ev[:, :-1]
            motion_mag = diffs.abs().sum(1, keepdim=True)

        # 2D fusion → img_feats 보강 (truck/car↑)
        img_feats_2d = self.fusion_2d(img_feats, F_ego, F_obj, motion_mag)

        # 3D cross-attention용 ev_feats
        B, N, C, Hf, Wf = img_feats[0].shape
        ego_r = F.interpolate(F_ego, (Hf,Wf), mode='bilinear', align_corners=False)
        obj_r = F.interpolate(F_obj, (Hf,Wf), mode='bilinear', align_corners=False)
        ev_feats = self.ev_proj(torch.cat([ego_r, obj_r], dim=1))

        return ev, F_ego, F_obj, img_feats_2d, ev_feats

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

        img_feats                              = self.extract_feat(img=img)
        ev, F_ego, F_obj, img_feats_2d, ev_feats = self._extract_event(
            metas, img_feats, dev
        )

        losses = {}
        # VoxFormerHeadEvent: 2D-enhanced feature + event 3D cross-attention
        outs = self.pts_bbox_head(
            img_feats_2d,   # 2D fusion된 RGB feature (truck↑)
            metas, target,
            ev_feats=ev_feats,  # 3D cross-attention용 (person↑)
        )
        losses.update(self.pts_bbox_head.training_step(outs, target, metas))
        losses.update(self.compute_aux_losses(F_ego, F_obj, metas, dev))
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kw):
        lq    = img.size(1)
        metas = [each[lq-1] for each in img_metas]
        img   = img[:,-1,...]
        dev   = img.device

        img_feats                              = self.extract_feat(img=img)
        ev, F_ego, F_obj, img_feats_2d, ev_feats = self._extract_event(
            metas, img_feats, dev
        )
        outs = self.pts_bbox_head(img_feats_2d, metas, target, ev_feats=ev_feats)
        res  = self.pts_bbox_head.validation_step(outs, target, metas)
        return [res] if isinstance(res, dict) else res