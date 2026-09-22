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
import os
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
class VoxFormerExp2(MVXTwoStageDetector):
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

        super(VoxFormerExp2,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        # Exp2: EventEncoder + FusionConv
        _C = 128  # FPN output dim
        self.event_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, _C, 1),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(_C * 2, _C, 1),
            nn.BatchNorm2d(_C),
            nn.ReLU(inplace=True),
        )
        print("[Exp2] EventEncoder + FusionConv initialized")

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


    def _load_event_voxel(self, event_paths, img_H, img_W, device):
        import numpy as np
        import torch.nn.functional as F
        T, ev_H, ev_W = 5, 352, 1216
        batch = []
        for path in event_paths:
            voxel = np.zeros((T, ev_H, ev_W), dtype=np.float32)
            if path and os.path.exists(path):
                ev = np.load(path)
                if ev.ndim == 2 and ev.shape[1] == 4:
                    x = ev[:,0].astype(np.int32); y = ev[:,1].astype(np.int32)
                    t = ev[:,2].astype(np.float64); p = ev[:,3].astype(np.float32)
                    ok = (x>=0)&(x<ev_W)&(y>=0)&(y<ev_H)
                    x,y,t,p = x[ok],y[ok],t[ok],p[ok]
                    if len(t) > 0:
                        tn = (t-t.min())/(t.max()-t.min()+1e-8)*(T-1)
                        pol = np.where(p>0,1.,-1.).astype(np.float32)
                        tf = tn.astype(np.int32); tc = tf+1
                        wc = (tn-tf).astype(np.float32); wf = 1.-wc
                        mf = tf<T
                        if mf.sum(): np.add.at(voxel,(tf[mf],y[mf],x[mf]),pol[mf]*wf[mf])
                        mc = tc<T
                        if mc.sum(): np.add.at(voxel,(tc[mc],y[mc],x[mc]),pol[mc]*wc[mc])
            batch.append(voxel)
        ev_t = torch.from_numpy(np.stack(batch,0)).float().to(device)
        import torch.nn.functional as F
        ev_t = F.interpolate(ev_t, (img_H, img_W), mode='bilinear', align_corners=False)
        return ev_t

    def _fuse_event_feats(self, img_feats, img_metas, img_H, img_W):
        import torch.nn.functional as F
        paths = [m.get('event_path', None) for m in img_metas]
        if all(p is None for p in paths):
            return img_feats
        ev_voxel = self._load_event_voxel(paths, img_H, img_W, img_feats[0].device)
        ev_feat  = self.event_encoder(ev_voxel)
        fused = []
        for feat in img_feats:
            B, N, C, Hf, Wf = feat.shape
            ev_r = F.interpolate(ev_feat, (Hf, Wf), mode='bilinear', align_corners=False)
            ev_r = ev_r.unsqueeze(1).expand(B, N, C, Hf, Wf)
            cat  = torch.cat([feat, ev_r], dim=2).view(B*N, 2*C, Hf, Wf)
            out  = self.fusion_conv(cat).view(B, N, C, Hf, Wf)
            fused.append(out)
        return fused

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
        img_feats = self._fuse_event_feats(img_feats, img_metas, img_H, img_W)
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
        img_feats = self._fuse_event_feats(img_feats, img_metas, img_H, img_W)
        # ─────────────────────────────────────────────────────

        outs = self.pts_bbox_head(img_feats, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        if isinstance(completion_results, dict):
            completion_results = [completion_results]

        if isinstance(completion_results, dict):
            completion_results = [completion_results]
        return completion_results
