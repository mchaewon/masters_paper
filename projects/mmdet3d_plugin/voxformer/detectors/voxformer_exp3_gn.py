"""
VoxFormerExp3 with GroupNorm
============================
voxformer_exp3.py와 동일하나 BatchNorm → GroupNorm으로 변경.
batch_size=1 환경에서 안정적인 학습을 위해 GroupNorm 사용.

GroupNorm 설정:
  num_groups=8 (모든 채널이 8의 배수: 16,32,64,128)
  Conv3D의 경우 GroupNorm이 3D 텐서에도 동작 (채널 차원만 사용)
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
# E_ego: Spatial encoder (GroupNorm 버전)
# ============================================================
class EgoEventEncoderGN(nn.Module):
    """
    Event voxel의 공간 분포 정보 추출.
    Bin 합산 후 Conv2D + GroupNorm.
    Auxiliary: depth prediction head
    """
    def __init__(self, out_channels=128, num_groups=8):
        super().__init__()
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            # (B, 1, H, W) → (B, 32, H, W)
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(inplace=True),
            # → (B, 64, H/2, W/2)
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
            # → (B, C, H/4, W/4)
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

        # Auxiliary depth head (training only)
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Softplus()   # depth > 0
        )

    def forward(self, event_voxel):
        """
        event_voxel: (B, num_bins, H, W)
        returns: feat (B, C, H/4, W/4)
        """
        spatial = event_voxel.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        feat = self.encoder(spatial)                      # (B, C, H/4, W/4)
        return feat

    def predict_depth(self, feat):
        return self.depth_head(feat)  # (B, 1, H, W)


# ============================================================
# E_obj: Temporal encoder (GroupNorm 버전)
# ============================================================
class ObjEventEncoderGN(nn.Module):
    """
    Event voxel의 시간 변화 패턴 추출.
    2배 Spatial Downsample 후 Conv3D + GroupNorm.
    Auxiliary: moving object segmentation head
    """
    def __init__(self, num_bins=5, out_channels=128, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        self.num_bins = num_bins

        # GroupNorm은 (B, C, *) 형태에서 C 차원 기준 동작
        # Conv3D 출력: (B, C, D, H, W) → GroupNorm(num_groups, C) 적용
        self.encoder = nn.Sequential(
            # (B, 1, bins, H/2, W/2) → (B, 16, bins, H/2, W/2)
            nn.Conv3d(1, 16, (3, 3, 3), padding=1, bias=False),
            nn.GroupNorm(4, 16),   # 16채널, groups=4
            nn.ReLU(inplace=True),
            # → (B, 32, bins, H/4, W/4)
            nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1),
                      stride=(1, 2, 2), bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(inplace=True),
            # → (B, 64, bins, H/4, W/4)
            nn.Conv3d(32, 64, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
        )

        # Temporal pooling + projection
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.proj = nn.Sequential(
            nn.Conv2d(64, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )  # (B, 64, H/4, W/4) → (B, C, H/4, W/4)

        # Auxiliary moving object head (training only)
        self.dynamic_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )  # (B, C, H/4, W/4) → (B, 1, H/2, W/2) → upsample → (B, 1, H, W)

    def forward(self, event_voxel):
        """
        event_voxel: (B, num_bins, H, W)
        returns: feat (B, C, H/4, W/4)
        """
        B, T, H, W = event_voxel.shape

        # Spatial downsample (VRAM 절감)
        ev_down = F.avg_pool2d(
            event_voxel.view(B * T, 1, H, W),
            kernel_size=2, stride=2
        ).view(B, T, H // 2, W // 2)       # (B, bins, H/2, W/2)

        x = ev_down.unsqueeze(1)            # (B, 1, bins, H/2, W/2)
        x = self.encoder(x)                 # (B, 64, bins, H/4, W/4)
        x = self.temporal_pool(x).squeeze(2)  # (B, 64, H/4, W/4)
        feat = self.proj(x)                 # (B, C, H/4, W/4)
        return feat

    def predict_dynamic(self, feat):
        return self.dynamic_head(feat)      # (B, 1, H/2, W/2)


# ============================================================
# Event Fusion Module (GroupNorm 버전)
# ============================================================
class EventFusionModuleGN(nn.Module):
    """
    RGB features + F_ego + F_obj → fused features
    1x1 Conv + GroupNorm
    """
    def __init__(self, rgb_channels=128, event_channels=128, num_groups=8):
        super().__init__()
        in_ch = rgb_channels + event_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, rgb_channels, 1, bias=False),
            nn.GroupNorm(num_groups, rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feats, F_ego, F_obj):
        """
        img_feats: list of [(B, N, C, H', W')]
        F_ego:     (B, C, H/4, W/4)
        F_obj:     (B, C, H/4, W/4)
        returns:   list of [(B, N, C, H', W')]
        """
        fused_feats = []
        for feat in img_feats:
            B, N, C, Hf, Wf = feat.shape

            # Event features → FPN 해상도로 resize
            ego_r = F.interpolate(
                F_ego, size=(Hf, Wf),
                mode='bilinear', align_corners=False
            )  # (B, C, Hf, Wf)
            obj_r = F.interpolate(
                F_obj, size=(Hf, Wf),
                mode='bilinear', align_corners=False
            )  # (B, C, Hf, Wf)

            # N_cam 차원으로 expand
            ego_r = ego_r.unsqueeze(1).expand(B, N, C, Hf, Wf)
            obj_r = obj_r.unsqueeze(1).expand(B, N, C, Hf, Wf)

            # concat → fusion conv
            cat = torch.cat([feat, ego_r, obj_r], dim=2)   # (B, N, 3C, Hf, Wf)
            cat = cat.view(B * N, 3 * C, Hf, Wf)
            out = self.fusion(cat)                           # (B*N, C, Hf, Wf)
            out = out.view(B, N, C, Hf, Wf)
            fused_feats.append(out)

        return fused_feats


# ============================================================
# Event Loading Utilities
# ============================================================
def load_event_voxel_batch(img_metas, num_bins=5, ev_H=352, ev_W=1216,
                            img_H=370, img_W=1220, device='cpu'):
    """
    미리 저장된 voxel npy 우선 로드.
    없으면 raw events 실시간 변환 (fallback).

    경로 패턴:
      events/{seq}/image_0/000000.npy
      → events_voxel/{seq}/image_0/000000.npy
    """
    batch = []
    for meta in img_metas:
        path = meta.get('event_path', None)
        voxel = None

        if path:
            # 미리 저장된 voxel 경로
            voxel_path = path.replace('/events/', '/events_voxel/')

            if os.path.exists(voxel_path):
                data = np.load(voxel_path)
                if data.ndim == 3:      # (bins, H, W)
                    voxel = data.astype(np.float32)

            # fallback: raw events 실시간 변환
            if voxel is None and os.path.exists(path):
                events = np.load(path)
                if events.ndim == 3:
                    voxel = events.astype(np.float32)
                elif events.ndim == 2 and events.shape[1] == 4:
                    voxel = _raw_events_to_voxel(events, num_bins, ev_H, ev_W)

        if voxel is None:
            voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)

        batch.append(voxel)

    ev_tensor = torch.from_numpy(
        np.stack(batch, axis=0)
    ).float().to(device)    # (B, bins, ev_H, ev_W)

    # VoxFormer 입력 해상도로 resize
    if ev_tensor.shape[2:] != (img_H, img_W):
        ev_tensor = F.interpolate(
            ev_tensor, size=(img_H, img_W),
            mode='bilinear', align_corners=False
        )
    return ev_tensor        # (B, bins, img_H, img_W)


def _raw_events_to_voxel(events, num_bins, ev_H, ev_W):
    """Raw events (N, 4) → voxel grid (bins, H, W)"""
    voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    t = events[:, 2].astype(np.float64)
    p = events[:, 3].astype(np.float32)
    ok = (x >= 0) & (x < ev_W) & (y >= 0) & (y < ev_H)
    x, y, t, p = x[ok], y[ok], t[ok], p[ok]
    if len(t) > 0:
        tn  = (t - t.min()) / (t.max() - t.min() + 1e-8) * (num_bins - 1)
        pol = np.where(p > 0, 1., -1.).astype(np.float32)
        tf  = tn.astype(np.int32)
        tc  = tf + 1
        wc  = (tn - tf).astype(np.float32)
        wf  = 1. - wc
        mf  = tf < num_bins
        mc  = tc < num_bins
        if mf.sum():
            np.add.at(voxel, (tf[mf], y[mf], x[mf]), pol[mf] * wf[mf])
        if mc.sum():
            np.add.at(voxel, (tc[mc], y[mc], x[mc]), pol[mc] * wc[mc])
    return voxel


def load_depth_batch(img_metas, img_H=370, img_W=1220,
                     max_depth=80.0, device='cpu'):
    """depth_path → (B, 1, img_H, img_W)"""
    batch = []
    for meta in img_metas:
        path = meta.get('depth_path', None)
        if path and os.path.exists(path):
            d = np.load(path).astype(np.float32)
            # crop to VoxFormer resolution
            dh, dw = d.shape
            if dh >= img_H and dw >= img_W:
                d = d[:img_H, :img_W]
            else:
                d = np.array(
                    __import__('PIL').Image.fromarray(d).resize(
                        (img_W, img_H), 2   # BILINEAR
                    )
                )
            d = np.clip(d, 0, max_depth)
        else:
            d = np.zeros((img_H, img_W), dtype=np.float32)
        batch.append(d)

    return torch.from_numpy(
        np.stack(batch)[:, np.newaxis]
    ).float().to(device)    # (B, 1, H, W)


def load_moving_mask_batch(img_metas, img_H=370, img_W=1220, device='cpu'):
    """moving_mask_path → (B, 1, img_H, img_W). 값: 1/0/-1"""
    batch = []
    for meta in img_metas:
        path = meta.get('moving_mask_path', None)
        if path and os.path.exists(path):
            m = np.load(path).astype(np.float32)
            if m.shape != (img_H, img_W):
                from PIL import Image as PILImage
                m_u = ((m + 1) * 64).astype(np.uint8)
                m   = np.array(
                    PILImage.fromarray(m_u).resize(
                        (img_W, img_H), PILImage.NEAREST
                    )
                ).astype(np.float32) / 64 - 1
        else:
            m = np.full((img_H, img_W), -1, dtype=np.float32)
        batch.append(m)

    return torch.from_numpy(
        np.stack(batch)[:, np.newaxis]
    ).float().to(device)    # (B, 1, H, W)


# ============================================================
# Main Detector
# ============================================================
@DETECTORS.register_module()
class VoxFormerExp3GN(MVXTwoStageDetector):
    """
    VoxFormerExp3 with GroupNorm.
    Dual Event Encoder (Ego + Obj) + Auxiliary Supervision.
    GroupNorm으로 batch_size=1 환경에서 안정적 학습.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None, pts_bbox_head=None,
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

        # Dual Event Encoder (GroupNorm)
        self.ego_encoder = EgoEventEncoderGN(
            out_channels=event_channels, num_groups=num_groups
        )
        self.obj_encoder = ObjEventEncoderGN(
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

        total_event_params = (
            sum(p.numel() for p in self.ego_encoder.parameters()) +
            sum(p.numel() for p in self.obj_encoder.parameters()) +
            sum(p.numel() for p in self.event_fusion.parameters())
        )
        print(f"[VoxFormerExp3GN] GroupNorm 버전 초기화 완료")
        print(f"  event branch params: {total_event_params:,}")
        print(f"  lambda_ego={lambda_ego}, lambda_obj={lambda_obj}")

    # ----------------------------------------------------------
    # Feature Extraction
    # ----------------------------------------------------------
    def extract_event_feats(self, img_metas, device):
        ev = load_event_voxel_batch(
            img_metas,
            num_bins=self.num_event_bins,
            ev_H=self.ev_H, ev_W=self.ev_W,
            img_H=self.img_H, img_W=self.img_W,
            device=device
        )  # (B, bins, H, W)
        F_ego = self.ego_encoder(ev)   # (B, C, H/4, W/4)
        F_obj = self.obj_encoder(ev)   # (B, C, H/4, W/4)
        return ev, F_ego, F_obj

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        losses = {}

        # ── Ego: depth L1 loss ───────────────────────────────
        depth_gt   = load_depth_batch(
            img_metas, self.img_H, self.img_W, device=device
        )   # (B, 1, H, W)
        depth_pred = self.ego_encoder.predict_depth(F_ego)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(
                depth_pred, size=depth_gt.shape[2:],
                mode='bilinear', align_corners=False
            )
        valid_d = (depth_gt > 0).float()
        if valid_d.sum() > 0:
            losses['loss_ego_depth'] = (
                F.l1_loss(depth_pred * valid_d, depth_gt * valid_d,
                          reduction='sum') / (valid_d.sum() + 1e-6)
            ) * self.lambda_ego
        else:
            losses['loss_ego_depth'] = depth_pred.sum() * 0.

        # ── Obj: moving mask BCE loss ─────────────────────────
        moving_gt  = load_moving_mask_batch(
            img_metas, self.img_H, self.img_W, device=device
        )   # (B, 1, H, W)
        dyn_pred   = self.obj_encoder.predict_dynamic(F_obj)
        if dyn_pred.shape != moving_gt.shape:
            dyn_pred = F.interpolate(
                dyn_pred, size=moving_gt.shape[2:],
                mode='bilinear', align_corners=False
            )
        valid_m    = (moving_gt >= 0).float()
        if valid_m.sum() > 0:
            gt_binary = (moving_gt >= 0.5).float()
            losses['loss_obj_dyn'] = (
                F.binary_cross_entropy_with_logits(
                    dyn_pred * valid_m, gt_binary * valid_m,
                    reduction='sum'
                ) / (valid_m.sum() + 1e-6)
            ) * self.lambda_obj
        else:
            losses['loss_obj_dyn'] = dyn_pred.sum() * 0.

        return losses

    # ----------------------------------------------------------
    # VoxFormer 기존 메서드 (변경 없음)
    # ----------------------------------------------------------
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
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B/len_queue), len_queue, int(BN/B), C, H, W)
                )
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN/B), C, H, W)
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
        _, F_ego, F_obj = self.extract_event_feats(img_metas, device)

        # 3. Fusion
        img_feats = self.event_fusion(img_feats, F_ego, F_obj)

        # 4. SSC loss
        losses     = dict()
        outs       = self.pts_bbox_head(img_feats, img_metas, target)
        losses_pts = self.pts_bbox_head.training_step(outs, target, img_metas)
        losses.update(losses_pts)

        # 5. Auxiliary losses
        losses.update(self.compute_aux_losses(F_ego, F_obj, img_metas, device))

        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kwargs):
        len_queue = img.size(1)
        img_metas = [each[len_queue - 1] for each in img_metas]
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