"""
VoxFormerExp3 with Event Temporal Decomposition (DIFF)
======================================================
개선 1: E_obj를 Conv3D → Temporal DIFF 방식으로 교체

핵심 변경:
  기존 ObjEventEncoderGN:
    입력: event_voxel (B, T, H, W) → Conv3D로 암묵적 학습
    문제: "무엇을 학습하는지" 불투명, 메모리 heavy

  개선 MotionEventEncoderGN:
    입력: DIFF = event_voxel[:,1:] - event_voxel[:,:-1]  (B, T-1, H, W)
           MAG  = DIFF.abs().sum(dim=1)                   (B, 1,   H, W)
           → concat: (B, T, H, W) = (B, 5, H, W)
    의미: DIFF는 bin 간 event 위치 이동 = object motion velocity
          배경 ego-motion: 이미지 전체에 일정한 패턴 (상대적으로 균일)
          동적 객체: 국소적으로 빠른 이동 → DIFF에서 두드러짐

물리적 근거:
  SUM  → 전체 event 밀도 → depth proxy  (ego-motion 강도 ∝ 1/depth)
  DIFF → 시간적 변화율  → velocity proxy (object motion이 ego-motion 패턴에서 이탈)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


# ============================================================
# E_ego: Spatial encoder (SUM, Exp3 GN과 동일)
# ============================================================
class EgoEventEncoderGN(nn.Module):
    """
    Event voxel의 공간 분포 정보 추출.
    Bin 합산(SUM) → ego-motion depth 신호.
    Auxiliary: depth prediction head
    """
    def __init__(self, out_channels=128, num_groups=8):
        super().__init__()
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Softplus()
        )

    def forward(self, event_voxel):
        # SUM: 시간 축 합산 → 공간 분포 (ego-motion proxy)
        spatial = event_voxel.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return self.encoder(spatial)                      # (B, C, H/4, W/4)

    def predict_depth(self, feat):
        return self.depth_head(feat)


# ============================================================
# 개선된 E_obj: Temporal DIFF encoder
# ============================================================
class MotionEventEncoderGN(nn.Module):
    """
    Event Temporal Decomposition: DIFF 기반 object motion 인코더.

    DIFF 계산:
      diffs = event_voxel[:, 1:] - event_voxel[:, :-1]
            = (B, T-1, H, W)  ← 각 bin 간 event 변화

      mag   = diffs.abs().sum(dim=1)
            = (B, 1, H, W)    ← 전체 motion 강도

      input = cat([diffs, mag])
            = (B, T, H, W)    = (B, 5, H, W) for T=5

    물리적 의미:
      - 배경 (정적 물체 위 ego-motion events):
        bin 0→4로 일정하게 이동 → diffs ≈ 일정한 ego-motion 패턴
        전체 이미지에 걸쳐 균일

      - 동적 물체 (자체 motion):
        ego-motion + 자체 속도 → diffs에서 배경 패턴에서 이탈
        특정 물체 경계 위치에서만 강한 DIFF 값

    Auxiliary: moving object segmentation head
    """
    def __init__(self, num_bins=5, out_channels=128, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        self.num_bins = num_bins
        in_channels = num_bins  # (T-1) diffs + 1 mag = T

        self.encoder = nn.Sequential(
            # 입력: (B, T, H/2, W/2) after spatial downsample
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )  # (B, T, H/2, W/2) → (B, C, H/8, W/8) → upsample → (B, C, H/4, W/4)

        # 출력 해상도를 E_ego와 맞추기 위한 upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )  # (B, C, H/8, W/8) → (B, C, H/4, W/4)

        # Auxiliary moving object head
        self.dynamic_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, event_voxel):
        """
        event_voxel: (B, num_bins, H, W)
        returns: feat (B, C, H/4, W/4)
        """
        B, T, H, W = event_voxel.shape

        # ── Temporal DIFF 계산 ─────────────────────────────
        diffs = event_voxel[:, 1:] - event_voxel[:, :-1]  # (B, T-1, H, W)
        mag   = diffs.abs().sum(dim=1, keepdim=True)        # (B, 1,   H, W)
        motion = torch.cat([diffs, mag], dim=1)             # (B, T,   H, W)

        # ── Spatial downsample (메모리 절감) ───────────────
        # (B, T, H, W) → (B, T, H/2, W/2)
        motion_down = F.avg_pool2d(
            motion.view(B * T, 1, H, W), kernel_size=2, stride=2
        ).view(B, T, H // 2, W // 2)

        # ── Encoding ───────────────────────────────────────
        feat = self.encoder(motion_down)    # (B, C, H/8, W/8)
        feat = self.upsample(feat)          # (B, C, H/4, W/4)
        return feat, mag, mag                                    # mag는 gate용으로 반환

    def predict_dynamic(self, feat):
        return self.dynamic_head(feat)      # (B, 1, H/2, W/2)


# ============================================================
# Event Fusion Module (Exp3 GN과 동일)
# ============================================================
class EventFusionModuleGN(nn.Module):
    def __init__(self, rgb_channels=128, event_channels=128, num_groups=8):
        super().__init__()
        in_ch = rgb_channels + event_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, rgb_channels, 1, bias=False),
            nn.GroupNorm(num_groups, rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feats, F_ego, F_obj):
        fused_feats = []
        for feat in img_feats:
            B, N, C, Hf, Wf = feat.shape
            ego_r = F.interpolate(F_ego, (Hf, Wf), mode='bilinear', align_corners=False)
            obj_r = F.interpolate(F_obj, (Hf, Wf), mode='bilinear', align_corners=False)
            ego_r = ego_r.unsqueeze(1).expand(B, N, C, Hf, Wf)
            obj_r = obj_r.unsqueeze(1).expand(B, N, C, Hf, Wf)
            cat   = torch.cat([feat, ego_r, obj_r], dim=2).view(B * N, 3 * C, Hf, Wf)
            out   = self.fusion(cat).view(B, N, C, Hf, Wf)
            fused_feats.append(out)
        return fused_feats


# ============================================================
# Utility: Event/Depth/Mask Loading
# ============================================================
def load_event_voxel_batch(img_metas, num_bins=5, ev_H=352, ev_W=1216,
                            img_H=370, img_W=1220, device='cpu'):
    batch = []
    for meta in img_metas:
        path  = meta.get('event_path', None)
        voxel = None
        if path:
            voxel_path = path.replace('/events/', '/events_voxel/')
            if os.path.exists(voxel_path):
                data = np.load(voxel_path)
                if data.ndim == 3:
                    voxel = data.astype(np.float32)
            if voxel is None and os.path.exists(path):
                events = np.load(path)
                if events.ndim == 3:
                    voxel = events.astype(np.float32)
                elif events.ndim == 2 and events.shape[1] == 4:
                    voxel = _raw_events_to_voxel(events, num_bins, ev_H, ev_W)
        if voxel is None:
            voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)
        batch.append(voxel)
    ev = torch.from_numpy(np.stack(batch)).float().to(device)
    if ev.shape[2:] != (img_H, img_W):
        ev = F.interpolate(ev, (img_H, img_W), mode='bilinear', align_corners=False)
    return ev


def _raw_events_to_voxel(events, num_bins, ev_H, ev_W):
    voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)
    x = events[:,0].astype(np.int32); y = events[:,1].astype(np.int32)
    t = events[:,2].astype(np.float64); p = events[:,3].astype(np.float32)
    ok = (x>=0)&(x<ev_W)&(y>=0)&(y<ev_H)
    x,y,t,p = x[ok],y[ok],t[ok],p[ok]
    if len(t) > 0:
        tn  = (t-t.min())/(t.max()-t.min()+1e-8)*(num_bins-1)
        pol = np.where(p>0,1.,-1.).astype(np.float32)
        tf  = tn.astype(np.int32); tc = tf+1
        wc  = (tn-tf).astype(np.float32); wf = 1.-wc
        mf  = tf<num_bins; mc = tc<num_bins
        if mf.sum(): np.add.at(voxel,(tf[mf],y[mf],x[mf]),pol[mf]*wf[mf])
        if mc.sum(): np.add.at(voxel,(tc[mc],y[mc],x[mc]),pol[mc]*wc[mc])
    return voxel


def load_depth_batch(img_metas, img_H=370, img_W=1220, max_depth=80., device='cpu'):
    batch = []
    for meta in img_metas:
        path = meta.get('depth_path', None)
        if path and os.path.exists(path):
            d = np.load(path).astype(np.float32)
            dh, dw = d.shape
            if dh >= img_H and dw >= img_W:
                d = d[:img_H, :img_W]
            d = np.clip(d, 0, max_depth)
        else:
            d = np.zeros((img_H, img_W), dtype=np.float32)
        batch.append(d)
    return torch.from_numpy(np.stack(batch)[:,np.newaxis]).float().to(device)


def load_moving_mask_batch(img_metas, img_H=370, img_W=1220, device='cpu'):
    batch = []
    for meta in img_metas:
        path = meta.get('moving_mask_path', None)
        if path and os.path.exists(path):
            m = np.load(path).astype(np.float32)
            if m.shape != (img_H, img_W):
                from PIL import Image as PILImage
                m_u = ((m+1)*64).astype(np.uint8)
                m   = np.array(
                    PILImage.fromarray(m_u).resize((img_W, img_H), PILImage.NEAREST)
                ).astype(np.float32)/64 - 1
        else:
            m = np.full((img_H, img_W), -1, dtype=np.float32)
        batch.append(m)
    return torch.from_numpy(np.stack(batch)[:,np.newaxis]).float().to(device)


# ============================================================
# Main Detector: VoxFormerExp3Diff
# ============================================================
@DETECTORS.register_module()
class VoxFormerExp3Diff(MVXTwoStageDetector):
    """
    Exp3 GN + Improvement 1: E_obj → Temporal DIFF encoder

    변경사항:
      ObjEventEncoderGN (Conv3D) → MotionEventEncoderGN (Temporal DIFF)

    나머지 (EgoEncoder, FusionModule, Auxiliary loss)는 Exp3 GN과 동일.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None, pts_bbox_head=None,
                 img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None,
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

        # E_ego: SUM 기반 (Exp3 GN과 동일)
        self.ego_encoder = EgoEventEncoderGN(
            out_channels=event_channels, num_groups=num_groups
        )

        # E_obj: DIFF 기반 (개선 1)
        self.obj_encoder = MotionEventEncoderGN(
            num_bins=num_event_bins,
            out_channels=event_channels,
            num_groups=num_groups
        )

        self.event_fusion = EventFusionModuleGN(
            rgb_channels=128,
            event_channels=event_channels,
            num_groups=num_groups
        )

        self.lambda_ego    = lambda_ego
        self.lambda_obj    = lambda_obj
        self.num_event_bins = num_event_bins
        self.img_H, self.img_W = img_H, img_W
        self.ev_H,  self.ev_W  = ev_H,  ev_W

        n_ego = sum(p.numel() for p in self.ego_encoder.parameters())
        n_obj = sum(p.numel() for p in self.obj_encoder.parameters())
        print(f"[VoxFormerExp3Diff] Temporal DIFF encoder initialized")
        print(f"  EgoEncoder(SUM):  {n_ego:,} params")
        print(f"  ObjEncoder(DIFF): {n_obj:,} params")

    def extract_event_feats(self, img_metas, device):
        ev = load_event_voxel_batch(
            img_metas, num_bins=self.num_event_bins,
            ev_H=self.ev_H, ev_W=self.ev_W,
            img_H=self.img_H, img_W=self.img_W, device=device
        )
        F_ego = self.ego_encoder(ev)   # (B, C, H/4, W/4)
        F_obj, _ = self.obj_encoder(ev)   # (B, C, H/4, W/4), mag 무시
        return ev, F_ego, F_obj

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        losses = {}

        # E_ego auxiliary: depth L1 loss
        depth_gt   = load_depth_batch(img_metas, self.img_H, self.img_W, device=device)
        depth_pred = self.ego_encoder.predict_depth(F_ego)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(depth_pred, depth_gt.shape[2:],
                                        mode='bilinear', align_corners=False)
        valid_d = (depth_gt > 0).float()
        if valid_d.sum() > 0:
            losses['loss_ego_depth'] = (
                F.l1_loss(depth_pred*valid_d, depth_gt*valid_d, reduction='sum')
                / (valid_d.sum() + 1e-6)
            ) * self.lambda_ego
        else:
            losses['loss_ego_depth'] = depth_pred.sum() * 0.

        # E_obj auxiliary: moving mask BCE loss
        moving_gt  = load_moving_mask_batch(img_metas, self.img_H, self.img_W, device=device)
        dyn_pred   = self.obj_encoder.predict_dynamic(F_obj)
        if dyn_pred.shape != moving_gt.shape:
            dyn_pred = F.interpolate(dyn_pred, moving_gt.shape[2:],
                                      mode='bilinear', align_corners=False)
        valid_m = (moving_gt >= 0).float()
        if valid_m.sum() > 0:
            gt_bin = (moving_gt >= 0.5).float()
            losses['loss_obj_dyn'] = (
                F.binary_cross_entropy_with_logits(
                    dyn_pred*valid_m, gt_bin*valid_m, reduction='sum'
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
        img_metas = [each[len_queue-1] for each in img_metas]
        img       = img[:, -1, ...]
        device    = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj = self.extract_event_feats(img_metas, device)
        img_feats = self.event_fusion(img_feats, F_ego, F_obj)

        losses = dict()
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        losses.update(self.pts_bbox_head.training_step(outs, target, img_metas))
        losses.update(self.compute_aux_losses(F_ego, F_obj, img_metas, device))
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kwargs):
        len_queue = img.size(1)
        img_metas = [each[len_queue-1] for each in img_metas]
        img       = img[:, -1, ...]
        device    = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj = self.extract_event_feats(img_metas, device)
        img_feats = self.event_fusion(img_feats, F_ego, F_obj)

        outs = self.pts_bbox_head(img_feats, img_metas, target)
        results = self.pts_bbox_head.validation_step(outs, target, img_metas)
        if isinstance(results, dict):
            results = [results]
        return results