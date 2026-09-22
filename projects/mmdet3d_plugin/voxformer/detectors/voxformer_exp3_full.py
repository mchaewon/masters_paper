"""
VoxFormerExp3Full: 개선 1+2+3 통합
  1. E_obj: Temporal DIFF (MotionEventEncoderGN)
  2. Moving Mask: dense v2 (moving_masks_v2/ 우선 사용)
  3. Fusion: Motion-conditioned Gated (MotionGatedFusion)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


# ── E_ego: SUM 기반 (Exp3 GN과 동일) ────────────────────────
class EgoEventEncoderGN(nn.Module):
    def __init__(self, out_channels=128, num_groups=8):
        super().__init__()
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

    def forward(self, ev):
        return self.encoder(ev.sum(1, keepdim=True))

    def predict_depth(self, feat):
        return self.depth_head(feat)


# ── 개선 1: E_obj → Temporal DIFF ───────────────────────────
class MotionEventEncoderGN(nn.Module):
    """
    DIFF = event[:,1:] - event[:,:-1]  (bin간 변화 = object motion)
    MAG  = DIFF.abs().sum(1)           (motion 강도, 개선 3 gate에 사용)
    input = cat([DIFF, MAG]) = (B, T, H, W)
    """
    def __init__(self, num_bins=5, out_channels=128, num_groups=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_bins, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )
        self.dyn_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, ev):
        B, T, H, W = ev.shape
        diffs  = ev[:, 1:] - ev[:, :-1]                   # (B, T-1, H, W)
        mag    = diffs.abs().sum(1, keepdim=True)           # (B, 1, H, W)
        motion = torch.cat([diffs, mag], dim=1)             # (B, T, H, W)
        # spatial downsample
        md = F.avg_pool2d(
            motion.view(B*T, 1, H, W), 2, 2
        ).view(B, T, H//2, W//2)
        feat = self.up(self.encoder(md))                    # (B, C, H/4, W/4)
        return feat, mag                                    # mag: gate용

    def predict_dynamic(self, feat):
        return self.dyn_head(feat)


# ── 개선 3: Motion-conditioned Gated Fusion ──────────────────
class MotionGatedFusion(nn.Module):
    """
    gate = σ(motion_magnitude) → dynamic 위치에서 F_obj 비중 자동 증가
    static 위치: (1-gate)*F_ego + gate*F_obj ≈ F_ego
    dynamic 위치:                             ≈ F_obj
    """
    def __init__(self, rgb_ch=128, ev_ch=128, num_groups=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(rgb_ch + ev_ch * 2, rgb_ch, 1, bias=False),
            nn.GroupNorm(num_groups, rgb_ch), nn.ReLU(inplace=True)
        )

    def forward(self, img_feats, F_ego, F_obj, motion_mag):
        out = []
        for feat in img_feats:
            B, N, C, Hf, Wf = feat.shape
            ego_r = F.interpolate(F_ego,      (Hf,Wf), mode='bilinear', align_corners=False)
            obj_r = F.interpolate(F_obj,      (Hf,Wf), mode='bilinear', align_corners=False)
            mag_r = F.interpolate(motion_mag, (Hf,Wf), mode='bilinear', align_corners=False)
            g     = self.gate(mag_r)                       # (B, 1, Hf, Wf)
            ego_w = ((1-g) * ego_r).unsqueeze(1).expand(B,N,C,Hf,Wf)
            obj_w = (   g  * obj_r).unsqueeze(1).expand(B,N,C,Hf,Wf)
            cat   = torch.cat([feat, ego_w, obj_w], 2).view(B*N, 3*C, Hf, Wf)
            out.append(self.proj(cat).view(B,N,C,Hf,Wf))
        return out


# ── Utilities ────────────────────────────────────────────────
def _to_voxel(events, T, ev_H, ev_W):
    v = np.zeros((T, ev_H, ev_W), dtype=np.float32)
    x = events[:,0].astype(np.int32); y = events[:,1].astype(np.int32)
    t = events[:,2].astype(np.float64); p = events[:,3].astype(np.float32)
    ok = (x>=0)&(x<ev_W)&(y>=0)&(y<ev_H)
    x,y,t,p = x[ok],y[ok],t[ok],p[ok]
    if len(t):
        tn = (t-t.min())/(t.max()-t.min()+1e-8)*(T-1)
        pol = np.where(p>0,1.,-1.).astype(np.float32)
        tf = tn.astype(np.int32); tc = tf+1
        wc = (tn-tf).astype(np.float32); wf = 1.-wc
        mf = tf<T; mc = tc<T
        if mf.sum(): np.add.at(v,(tf[mf],y[mf],x[mf]),pol[mf]*wf[mf])
        if mc.sum(): np.add.at(v,(tc[mc],y[mc],x[mc]),pol[mc]*wc[mc])
    return v


def load_event_voxel_batch(metas, T=5, ev_H=352, ev_W=1216,
                            img_H=370, img_W=1220, device='cpu'):
    batch = []
    for m in metas:
        path = m.get('event_path', None); vox = None
        if path:
            vp = path.replace('/events/', '/events_voxel/')
            if os.path.exists(vp):
                d = np.load(vp)
                if d.ndim == 3: vox = d.astype(np.float32)
            if vox is None and os.path.exists(path):
                d = np.load(path)
                if d.ndim == 3: vox = d.astype(np.float32)
                elif d.ndim == 2 and d.shape[1] == 4:
                    vox = _to_voxel(d, T, ev_H, ev_W)
        if vox is None: vox = np.zeros((T, ev_H, ev_W), np.float32)
        batch.append(vox)
    ev = torch.from_numpy(np.stack(batch)).float().to(device)
    if ev.shape[2:] != (img_H, img_W):
        ev = F.interpolate(ev, (img_H, img_W), mode='bilinear', align_corners=False)
    return ev


def load_depth_batch(metas, H=370, W=1220, mx=80., device='cpu'):
    batch = []
    for m in metas:
        p = m.get('depth_path', None)
        if p and os.path.exists(p):
            d = np.load(p).astype(np.float32)
            d = d[:H, :W] if d.shape[0]>=H and d.shape[1]>=W else d
            d = np.clip(d, 0, mx)
        else:
            d = np.zeros((H, W), np.float32)
        batch.append(d)
    return torch.from_numpy(np.stack(batch)[:,None]).float().to(device)


def load_moving_mask_batch(metas, H=370, W=1220, device='cpu'):
    """개선 2: moving_masks_v2 경로 우선"""
    batch = []
    for m in metas:
        path = m.get('moving_mask_path', None); arr = None
        if path:
            p2 = path.replace('/moving_masks/', '/moving_masks_v2/')
            for pp in [p2, path]:
                if os.path.exists(pp):
                    arr = np.load(pp).astype(np.float32); break
        if arr is None:
            arr = np.full((H, W), -1, np.float32)
        elif arr.shape != (H, W):
            from PIL import Image as PILImage
            arr = np.array(
                PILImage.fromarray(((arr+1)*64).astype(np.uint8))
                         .resize((W,H), PILImage.NEAREST)
            ).astype(np.float32)/64 - 1
        batch.append(arr)
    return torch.from_numpy(np.stack(batch)[:,None]).float().to(device)


# ── Main Detector ─────────────────────────────────────────────
@DETECTORS.register_module()
class VoxFormerExp3Full(MVXTwoStageDetector):
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
                 img_H=370, img_W=1220, ev_H=352, ev_W=1216):
        super().__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck,
            pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)

        self.ego_encoder  = EgoEventEncoderGN(event_channels, num_groups)
        self.obj_encoder  = MotionEventEncoderGN(num_event_bins, event_channels, num_groups)
        self.event_fusion = MotionGatedFusion(128, event_channels, num_groups)

        self.lambda_ego = lambda_ego; self.lambda_obj = lambda_obj
        self.T = num_event_bins
        self.img_H, self.img_W = img_H, img_W
        self.ev_H,  self.ev_W  = ev_H,  ev_W
        print(f"[VoxFormerExp3Full] Improvement 1+2+3 initialized")

    def _ev(self, metas, device):
        ev            = load_event_voxel_batch(metas, self.T, self.ev_H, self.ev_W,
                                               self.img_H, self.img_W, device)
        F_ego         = self.ego_encoder(ev)
        F_obj, mag    = self.obj_encoder(ev)
        return ev, F_ego, F_obj, mag

    def _aux(self, F_ego, F_obj, metas, device):
        losses = {}
        # depth L1
        dgt  = load_depth_batch(metas, self.img_H, self.img_W, device=device)
        dpred = self.ego_encoder.predict_depth(F_ego)
        if dpred.shape != dgt.shape:
            dpred = F.interpolate(dpred, dgt.shape[2:], mode='bilinear', align_corners=False)
        vd = (dgt > 0).float()
        losses['loss_ego_depth'] = (
            F.l1_loss(dpred*vd, dgt*vd, reduction='sum') / (vd.sum()+1e-6)
        ) * self.lambda_ego if vd.sum() > 0 else dpred.sum()*0.
        # moving BCE
        mgt   = load_moving_mask_batch(metas, self.img_H, self.img_W, device=device)
        mpred = self.obj_encoder.predict_dynamic(F_obj)
        if mpred.shape != mgt.shape:
            mpred = F.interpolate(mpred, mgt.shape[2:], mode='bilinear', align_corners=False)
        vm = (mgt >= 0).float()
        losses['loss_obj_dyn'] = (
            F.binary_cross_entropy_with_logits(
                mpred*vm, (mgt>=0.5).float()*vm, reduction='sum'
            ) / (vm.sum()+1e-6)
        ) * self.lambda_obj if vm.sum() > 0 else mpred.sum()*0.
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
        lq = img.size(1)
        metas = [e[lq-1] for e in img_metas]
        img   = img[:,  -1, ...]
        dev   = img.device
        feats = self.extract_feat(img=img)
        _, F_ego, F_obj, mag = self._ev(metas, dev)
        feats = self.event_fusion(feats, F_ego, F_obj, mag)
        losses = {}
        outs = self.pts_bbox_head(feats, metas, target)
        losses.update(self.pts_bbox_head.training_step(outs, target, metas))
        losses.update(self._aux(F_ego, F_obj, metas, dev))
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kw):
        lq = img.size(1)
        metas = [e[lq-1] for e in img_metas]
        img   = img[:, -1, ...]
        dev   = img.device
        feats = self.extract_feat(img=img)
        _, F_ego, F_obj, mag = self._ev(metas, dev)
        feats = self.event_fusion(feats, F_ego, F_obj, mag)
        outs  = self.pts_bbox_head(feats, metas, target)
        res   = self.pts_bbox_head.validation_step(outs, target, metas)
        return [res] if isinstance(res, dict) else res