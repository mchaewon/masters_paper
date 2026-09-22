"""
VoxFormerExp3_3D: Exp3 Diff + 3D Geometry-Aware Event Sampling
===============================================================
Exp3 Diff 대비 변경사항:
  - 2D level fusion (EventFusionModuleGN) 제거
  - ev_feats를 VoxFormerHeadEvent에 전달
  - Head에서 동일한 deformable cross-attention으로 3D sampling

흐름:
  RGB  → ResNet+FPN → mlvl_feats ─────────────────────────────┐
  Event → E_ego + E_obj(DIFF) → ev_feats ──────────────────────┤
                                                                 ↓
                              VoxFormerHeadEvent.forward(mlvl_feats, ev_feats)
                                  RGB cross-attn  → seed_rgb (N_q, C)
                                  Event cross-attn → seed_ev  (N_q, C)
                                  GatedFusion3D   → seed_fused
                                                                 ↓
                              self-attn diffusion → SSC head
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

# Exp3 Diff의 encoder 재사용
from .voxformer_exp3_diff import (
    EgoEventEncoderGN,
    MotionEventEncoderGN,
    load_event_voxel_batch,
    load_depth_batch,
    load_moving_mask_batch,
)


@DETECTORS.register_module()
class VoxFormerExp3_3D(MVXTwoStageDetector):
    """
    Exp3 Diff + 3D geometry-aware event sampling.

    pts_bbox_head: VoxFormerHeadEvent 사용 필수.

    vs Exp3 Diff:
      Exp3 Diff: ev_feats를 2D에서 RGB와 섞은 후 Head에 전달
                 → 3D lifting 시 geometry 정보 손실
      Exp3 3D:   ev_feats를 Head에 분리 전달
                 → Head 내부에서 동일한 deformable attention으로
                    geometry-consistent 3D sampling
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None,
                 pts_bbox_head=None,   # VoxFormerHeadEvent 사용
                 img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 event_channels=128,
                 num_event_bins=5,
                 num_groups=8,
                 lambda_ego=0.1,
                 lambda_obj=0.1,
                 img_H=370, img_W=1220,
                 ev_H=352, ev_W=1216,
                 ablation=None,  # None/'ego_only'/'obj_only'
                 ):
        super().__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck,
            pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )

        # E_ego: SUM (Exp3 Diff와 동일)
        self.ego_encoder = EgoEventEncoderGN(event_channels, num_groups)
        # E_obj: DIFF (Exp3 Diff와 동일)
        self.obj_encoder = MotionEventEncoderGN(
            num_event_bins, event_channels, num_groups
        )

        # 2D fusion 없음! ev_feats를 Head에 직접 전달
        # ev_feats projection: cat(F_ego, F_obj) → C채널로 통일
        self.ev_proj = nn.Sequential(
            nn.Conv2d(event_channels * 2, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels),
            nn.ReLU(inplace=True),
        )

        self.lambda_ego = lambda_ego
        self.lambda_obj = lambda_obj
        self.T   = num_event_bins
        self.ablation = ablation  # ablation 모드
        if ablation:
            print(f"[VoxFormerExp3_3D] Ablation mode: {ablation}")
        self.img_H, self.img_W = img_H, img_W
        self.ev_H,  self.ev_W  = ev_H,  ev_W

        print(f"[VoxFormerExp3_3D] 3D geometry-aware event sampling")
        print(f"  → pts_bbox_head에 VoxFormerHeadEvent 사용 필수")

    def get_ev_feats(self, img_metas, img_feats_shape, device):
        """
        Event encoder 실행 → ev_feats (B, C, H', W') 반환
        H', W'는 FPN output 해상도에 맞춤
        """
        ev = load_event_voxel_batch(
            img_metas, self.T,
            self.ev_H, self.ev_W,
            self.img_H, self.img_W, device
        )  # (B, T, H, W)

        F_ego   = self.ego_encoder(ev)       # (B, C, H/4, W/4)

        # Ablation: 특정 branch를 zero로 만들어 기여도 측정
        if self.ablation == 'obj_only':
            F_ego = torch.zeros_like(F_ego)   # E_ego 비활성화
        elif self.ablation == 'ego_only':
            pass  # E_obj는 아래서 처리
        obj_out = self.obj_encoder(ev)
        # 반환값이 tuple이든 tensor든 첫 번째 값(feat)만 사용
        F_obj   = obj_out[0] if isinstance(obj_out, (tuple, list)) else obj_out

        # Ablation: E_obj 비활성화
        if self.ablation == 'ego_only':
            F_obj = torch.zeros_like(F_obj)

        # FPN output 해상도로 맞추기
        B, N, C, Hf, Wf = img_feats_shape
        F_ego_r = F.interpolate(F_ego, (Hf, Wf), mode='bilinear',
                                 align_corners=False)
        F_obj_r = F.interpolate(F_obj, (Hf, Wf), mode='bilinear',
                                 align_corners=False)

        # E_ego + E_obj concat → projection → (B, C, Hf, Wf)
        ev_feats = self.ev_proj(torch.cat([F_ego_r, F_obj_r], dim=1))

        return ev, F_ego, F_obj, ev_feats

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        losses = {}
        # depth L1 (E_ego)
        depth_gt   = load_depth_batch(img_metas, self.img_H, self.img_W,
                                       device=device)
        depth_pred = self.ego_encoder.predict_depth(F_ego)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(depth_pred, depth_gt.shape[2:],
                                        mode='bilinear', align_corners=False)
        vd = (depth_gt > 0).float()
        losses['loss_ego_depth'] = (
            F.l1_loss(depth_pred*vd, depth_gt*vd, reduction='sum')
            / (vd.sum() + 1e-6)
        ) * self.lambda_ego if vd.sum() > 0 else depth_pred.sum() * 0.

        # moving mask BCE (E_obj)
        moving_gt = load_moving_mask_batch(img_metas, self.img_H, self.img_W,
                                            device=device)
        dyn_pred  = self.obj_encoder.predict_dynamic(F_obj)
        if dyn_pred.shape != moving_gt.shape:
            dyn_pred = F.interpolate(dyn_pred, moving_gt.shape[2:],
                                      mode='bilinear', align_corners=False)
        vm = (moving_gt >= 0).float()
        losses['loss_obj_dyn'] = (
            F.binary_cross_entropy_with_logits(
                dyn_pred*vm, (moving_gt>=0.5).float()*vm, reduction='sum'
            ) / (vm.sum() + 1e-6)
        ) * self.lambda_obj if vm.sum() > 0 else dyn_pred.sum() * 0.
        return losses

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            feats = self.img_backbone(img)
            if isinstance(feats, dict): feats = list(feats.values())
        else:
            return None
        if self.with_img_neck: feats = self.img_neck(feats)
        out = []
        for f in feats:
            BN, C, H, W = f.size()
            if len_queue:
                out.append(f.view(int(B/len_queue), len_queue,
                                   int(BN/B), C, H, W))
            else:
                out.append(f.view(B, int(BN/B), C, H, W))
        return out

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue)

    def forward(self, return_loss=True, **kwargs):
        return self.forward_train(**kwargs) if return_loss \
               else self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img_metas=None, img=None, target=None):
        lq     = img.size(1)
        metas  = [each[lq-1] for each in img_metas]
        img    = img[:, -1, ...]
        device = img.device

        # 1. RGB features
        img_feats = self.extract_feat(img=img)

        # 2. Event features (2D fusion 없이 분리 유지)
        _, F_ego, F_obj, ev_feats = self.get_ev_feats(
            metas, img_feats[0].shape, device
        )

        # 3. VoxFormerHeadEvent: RGB + Event 각각 3D cross-attention
        losses = {}
        outs   = self.pts_bbox_head(
            img_feats, metas, target,
            ev_feats=ev_feats,  # ← Head에 분리 전달
        )
        losses.update(
            self.pts_bbox_head.training_step(outs, target, metas)
        )

        # 4. Auxiliary losses
        losses.update(self.compute_aux_losses(F_ego, F_obj, metas, device))
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kwargs):
        lq     = img.size(1)
        metas  = [each[lq-1] for each in img_metas]
        img    = img[:, -1, ...]
        device = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj, ev_feats = self.get_ev_feats(
            metas, img_feats[0].shape, device
        )
        outs = self.pts_bbox_head(
            img_feats, metas, target,
            ev_feats=ev_feats,
        )
        res = self.pts_bbox_head.validation_step(outs, target, metas)
        if isinstance(res, dict): res = [res]
        return res