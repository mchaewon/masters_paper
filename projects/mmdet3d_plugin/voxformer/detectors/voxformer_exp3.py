"""
VoxFormerExp3: Dual Event Encoder with Auxiliary Supervision
============================================================
구조:
  Event Voxel (B, H, W)
       ├── E_ego (Conv2D on sum) → F_ego → [DepthHead → L_ego]
       └── E_obj (Conv3D)        → F_obj → [DynamicHead → L_obj]

  RGB feat + F_ego + F_obj → fusion → 기존 VoxFormer lifting → SSC

이 파일을 voxformer/detectors/ 에 추가하고
detectors/__init__.py에 등록한다.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


# ============================================================
# E_ego: Spatial encoder (sum of bins → depth-aware feature)
# ============================================================
class EgoEventEncoder(nn.Module):
    """
    Event voxel의 공간 분포 정보를 추출.
    Bin 합산 후 2D Conv으로 ego-motion 패턴 학습.
    Auxiliary: depth prediction head
    """
    def __init__(self, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        # Spatial feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )  # (1,H,W) → (C, H/4, W/4)

        # Auxiliary depth head (training only)
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Softplus()  # depth > 0
        )  # (C,H/4,W/4) → (1,H,W)

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
# E_obj: Temporal encoder (3D Conv → object-motion feature)
# ============================================================
class ObjEventEncoder(nn.Module):
    """
    Event voxel의 시간 변화 패턴을 추출.
    3D Conv으로 bin 간 motion 패턴 학습.
    Auxiliary: moving object segmentation head
    """
    def __init__(self, num_bins=5, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        # Temporal feature extractor (3D Conv on bins)
        self.encoder = nn.Sequential(
            # (B, 1, bins, H, W)
            nn.Conv3d(1, 16, (3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3,3,3), padding=(1,1,1),
                      stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, (3,1,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )  # (B,1,bins,H,W) → (B,64,bins,H/2,W/2)

        # Temporal pooling + projection
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.proj = nn.Sequential(
            nn.Conv2d(64, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )  # (B,64,H/2,W/2) → (B,C,H/2,W/2)

        # Auxiliary moving object head (training only)
        # binary: moving(1) vs static(0)
        self.dynamic_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )  # (B,C,H/2,W/2) → (B,1,H,W)

    def forward(self, event_voxel):
        """
        event_voxel: (B, num_bins, H, W)
        returns: feat (B, C, H/4, W/4)
        """
        # Conv3D 전에 spatial downsample → VRAM 절감
        B, T, H, W = event_voxel.shape
        ev_down = F.avg_pool2d(
            event_voxel.view(B*T, 1, H, W), kernel_size=2, stride=2
        ).view(B, T, H//2, W//2)          # (B,bins,H/2,W/2)

        x = ev_down.unsqueeze(1)           # (B,1,bins,H/2,W/2)
        x = self.encoder(x)                # (B,64,bins,H/4,W/4)
        x = self.temporal_pool(x).squeeze(2)  # (B,64,H/4,W/4)
        feat = self.proj(x)                # (B,C,H/4,W/4)
        return feat

    def predict_dynamic(self, feat):
        return self.dynamic_head(feat)  # (B,1,H,W)


# ============================================================
# Event Fusion Module
# ============================================================
class EventFusionModule(nn.Module):
    """
    RGB features + F_ego + F_obj → fused features
    img_feats shape: (B, N_cam, C, H', W')
    F_ego: (B, C, H/4, W/4) → resize to (H', W')
    F_obj: (B, C, H/2, W/2) → resize to (H', W')
    """
    def __init__(self, rgb_channels=128, event_channels=128):
        super().__init__()
        in_ch = rgb_channels + event_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feats, F_ego, F_obj):
        """
        img_feats: list of [(B, N, C, H', W')]
        returns:   list of [(B, N, C, H', W')]
        """
        fused_feats = []
        for feat in img_feats:
            B, N, C, Hf, Wf = feat.shape

            # event features를 FPN 해상도에 맞게 resize
            ego_r = F.interpolate(
                F_ego, size=(Hf, Wf),
                mode='bilinear', align_corners=False
            )  # (B, C, Hf, Wf)
            obj_r = F.interpolate(
                F_obj, size=(Hf, Wf),
                mode='bilinear', align_corners=False
            )  # (B, C, Hf, Wf)

            # N_cam 차원으로 확장
            ego_r = ego_r.unsqueeze(1).expand(B, N, C, Hf, Wf)
            obj_r = obj_r.unsqueeze(1).expand(B, N, C, Hf, Wf)

            # concat → fusion conv
            cat = torch.cat([feat, ego_r, obj_r], dim=2)  # (B,N,3C,Hf,Wf)
            cat = cat.view(B * N, 3 * C, Hf, Wf)
            out = self.fusion(cat)                          # (B*N, C, Hf, Wf)
            out = out.view(B, N, C, Hf, Wf)
            fused_feats.append(out)

        return fused_feats


# ============================================================
# Event Loading Utilities
# ============================================================
def load_event_voxel_batch(img_metas, num_bins=5, ev_H=352, ev_W=1216,
                            img_H=370, img_W=1220, device='cpu'):
    """
    미리 저장된 voxel npy를 로드.
    event_path가 raw events면 voxel_path로 변환하여 시도.
    """
    batch = []
    for meta in img_metas:
        path = meta.get('event_path', None)
        voxel = None

        if path:
            # 1. 미리 저장된 voxel 경로 시도
            # events/sequences/00/000000.npy
            # → events_voxel/sequences/00/events_voxel/000000.npy
            voxel_path = path.replace('/events/', '/events_voxel/')
            voxel_path = voxel_path.replace('/image_0/', '/events_voxel/')

            if os.path.exists(voxel_path):
                data = np.load(voxel_path)
                if data.ndim == 3:  # (B, H, W) voxel
                    voxel = data.astype(np.float32)
            
            # 2. voxel 없으면 raw events 실시간 변환 (fallback)
            if voxel is None and os.path.exists(path):
                events = np.load(path)
                if events.ndim == 3:
                    voxel = events.astype(np.float32)
                elif events.ndim == 2 and events.shape[1] == 4:
                    voxel = _raw_events_to_voxel(
                        events, num_bins, ev_H, ev_W
                    )

        if voxel is None:
            voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)

        batch.append(voxel)

    ev_tensor = torch.from_numpy(
        np.stack(batch, axis=0)
    ).float().to(device)

    if ev_tensor.shape[2:] != (img_H, img_W):
        ev_tensor = F.interpolate(
            ev_tensor, size=(img_H, img_W),
            mode='bilinear', align_corners=False
        )
    return ev_tensor


def _raw_events_to_voxel(events, num_bins, ev_H, ev_W):
    """Raw events (N,4) → voxel grid (B,H,W)"""
    voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)
    x = events[:,0].astype(np.int32); y = events[:,1].astype(np.int32)
    t = events[:,2].astype(np.float64); p = events[:,3].astype(np.float32)
    ok = (x>=0)&(x<ev_W)&(y>=0)&(y<ev_H)
    x,y,t,p = x[ok],y[ok],t[ok],p[ok]
    if len(t) > 0:
        tn = (t-t.min())/(t.max()-t.min()+1e-8)*(num_bins-1)
        pol = np.where(p>0,1.,-1.).astype(np.float32)
        tf = tn.astype(np.int32); tc = tf+1
        wc = (tn-tf).astype(np.float32); wf = 1.-wc
        mf = tf<num_bins
        if mf.sum(): np.add.at(voxel,(tf[mf],y[mf],x[mf]),pol[mf]*wf[mf])
        mc = tc<num_bins
        if mc.sum(): np.add.at(voxel,(tc[mc],y[mc],x[mc]),pol[mc]*wc[mc])
    return voxel


def load_depth_batch(img_metas, img_H=370, img_W=1220,
                     max_depth=80.0, device='cpu'):
    """depth_path에서 depth GT 로드 → (B, 1, img_H, img_W)"""
    batch = []
    for meta in img_metas:
        path = meta.get('depth_path', None)
        if path and os.path.exists(path):
            d = np.load(path).astype(np.float32)
            # crop to VoxFormer resolution
            d = d[:img_H, :img_W] if d.shape[0] >= img_H else d
            if d.shape != (img_H, img_W):
                d = np.array(
                    Image.fromarray(d).resize(
                        (img_W, img_H), Image.BILINEAR
                    )
                )
            d = np.clip(d, 0, max_depth)
        else:
            d = np.zeros((img_H, img_W), dtype=np.float32)
        batch.append(d)

    return torch.from_numpy(
        np.stack(batch)[:, np.newaxis]
    ).float().to(device)  # (B, 1, H, W)


def load_moving_mask_batch(img_metas, img_H=370, img_W=1220, device='cpu'):
    """
    moving_mask_path에서 mask 로드 → (B, 1, img_H, img_W)
    값: 1=moving, 0=static, -1=unknown
    """
    batch = []
    for meta in img_metas:
        path = meta.get('moving_mask_path', None)
        if path and os.path.exists(path):
            m = np.load(path).astype(np.float32)
            if m.shape != (img_H, img_W):
                # nearest neighbor (label 보존)
                from PIL import Image as PILImage
                m_uint = ((m + 1) * 64).astype(np.uint8)  # [-1,0,1] → [0,64,128]
                m_r = np.array(
                    PILImage.fromarray(m_uint).resize(
                        (img_W, img_H), PILImage.NEAREST
                    )
                ).astype(np.float32)
                m = m_r / 64 - 1
        else:
            m = np.full((img_H, img_W), -1, dtype=np.float32)
        batch.append(m)

    return torch.from_numpy(
        np.stack(batch)[:, np.newaxis]
    ).float().to(device)  # (B, 1, H, W)


# ============================================================
# Main Detector
# ============================================================
@DETECTORS.register_module()
class VoxFormerExp3(MVXTwoStageDetector):
    """
    VoxFormer + Dual Event Encoder (Ego + Obj)
    with Auxiliary Depth & Dynamic Supervision
    """

    def __init__(self,
                 # 기존 VoxFormer 파라미터
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None, pts_bbox_head=None,
                 img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 # Exp3 추가 파라미터
                 event_channels=128,
                 num_event_bins=5,
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

        # Dual Event Encoder
        self.ego_encoder = EgoEventEncoder(out_channels=event_channels)
        self.obj_encoder = ObjEventEncoder(
            num_bins=num_event_bins, out_channels=event_channels
        )

        # Fusion module (RGB C=128 fixed in VoxFormer)
        self.event_fusion = EventFusionModule(
            rgb_channels=128, event_channels=event_channels
        )

        # Loss weights
        self.lambda_ego = lambda_ego
        self.lambda_obj = lambda_obj

        # Image resolution
        self.img_H = img_H
        self.img_W = img_W
        self.ev_H  = ev_H
        self.ev_W  = ev_W

        print(f"[VoxFormerExp3] Dual Event Encoder initialized")
        print(f"  lambda_ego={lambda_ego}, lambda_obj={lambda_obj}")

    # ----------------------------------------------------------
    # Event Feature Extraction + Fusion
    # ----------------------------------------------------------
    def extract_event_feats(self, img_metas, device):
        """Event voxel → F_ego, F_obj"""
        ev = load_event_voxel_batch(
            img_metas,
            num_bins=self.obj_encoder.out_channels,  # placeholder
            ev_H=self.ev_H, ev_W=self.ev_W,
            img_H=self.img_H, img_W=self.img_W,
            device=device
        )  # (B, bins, H, W)

        F_ego = self.ego_encoder(ev)   # (B, C, H/4, W/4)
        F_obj = self.obj_encoder(ev)   # (B, C, H/2, W/2)
        return ev, F_ego, F_obj

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        """Auxiliary supervision: depth loss + dynamic mask loss"""
        losses = {}

        # ── Ego loss (depth prediction) ──────────────────────
        depth_gt = load_depth_batch(
            img_metas, self.img_H, self.img_W, device=device
        )  # (B, 1, H, W)

        depth_pred = self.ego_encoder.predict_depth(F_ego)  # (B, 1, H, W)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(
                depth_pred, size=depth_gt.shape[2:],
                mode='bilinear', align_corners=False
            )
        # valid pixel mask (depth > 0)
        valid_depth = (depth_gt > 0).float()
        if valid_depth.sum() > 0:
            losses['loss_ego_depth'] = (
                F.l1_loss(depth_pred * valid_depth,
                          depth_gt   * valid_depth, reduction='sum')
                / (valid_depth.sum() + 1e-6)
            ) * self.lambda_ego
        else:
            losses['loss_ego_depth'] = depth_pred.sum() * 0.0

        # ── Obj loss (moving mask prediction) ────────────────
        moving_gt = load_moving_mask_batch(
            img_metas, self.img_H, self.img_W, device=device
        )  # (B, 1, H, W), values: 1/0/-1

        dyn_pred = self.obj_encoder.predict_dynamic(F_obj)  # (B, 1, H, W)
        if dyn_pred.shape != moving_gt.shape:
            dyn_pred = F.interpolate(
                dyn_pred, size=moving_gt.shape[2:],
                mode='bilinear', align_corners=False
            )
        # valid: LiDAR가 닿은 픽셀만 (mask != -1)
        valid_dyn = (moving_gt >= 0).float()
        if valid_dyn.sum() > 0:
            gt_binary = (moving_gt >= 0.5).float()
            losses['loss_obj_dyn'] = (
                F.binary_cross_entropy_with_logits(
                    dyn_pred * valid_dyn,
                    gt_binary * valid_dyn,
                    reduction='sum'
                ) / (valid_dyn.sum() + 1e-6)
            ) * self.lambda_obj
        else:
            losses['loss_obj_dyn'] = dyn_pred.sum() * 0.0

        return losses

    # ----------------------------------------------------------
    # Forward (기존 VoxFormer와 동일 인터페이스)
    # ----------------------------------------------------------
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """기존 VoxFormer의 extract_img_feat 그대로"""
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
                    img_feat.view(int(B/len_queue), len_queue,
                                  int(BN/B), C, H, W)
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
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img_metas=None, img=None, target=None):
        len_queue   = img.size(1)
        img_metas   = [each[len_queue-1] for each in img_metas]
        img         = img[:, -1, ...]
        device      = img.device

        # 1. RGB feature extraction
        img_feats = self.extract_feat(img=img)

        # 2. Event feature extraction
        _, F_ego, F_obj = self.extract_event_feats(img_metas, device)

        # 3. Fusion
        img_feats = self.event_fusion(img_feats, F_ego, F_obj)

        # 4. SSC loss (기존)
        losses = dict()
        losses_pts = self.pts_bbox_head(img_feats, img_metas, target)
        losses_pts = self.pts_bbox_head.training_step(
            losses_pts, target, img_metas
        )
        losses.update(losses_pts)

        # 5. Auxiliary losses
        aux_losses = self.compute_aux_losses(F_ego, F_obj, img_metas, device)
        losses.update(aux_losses)

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
        completion_results = self.pts_bbox_head.validation_step(
            outs, target, img_metas
        )
        if isinstance(completion_results, dict):
            completion_results = [completion_results]
        return completion_results