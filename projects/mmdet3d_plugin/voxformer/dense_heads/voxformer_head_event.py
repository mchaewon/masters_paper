"""
VoxFormerHeadEvent: 3D Geometry-Aware Event Sampling
=====================================================
기존 VoxFormerHead에서 get_vox_features를 두 번 실행:
  1. RGB  features → seed_feats_rgb  (기존과 동일)
  2. Event features → seed_feats_ev  (동일한 3D→2D 투영 사용!)

핵심: 동일한 deformable cross-attention이 동일한 카메라 투영 행렬로
      RGB와 Event를 각각 샘플링 → 3D geometry-consistent

vs 기존 2D fusion (Exp3 Diff):
  2D interpolate로 event를 RGB에 섞은 후 lifting
  → geometry 정보 손실, event가 어느 3D 위치인지 모름

vs 이 방법:
  3D query position → 카메라 투영 → event map에서 정확한 위치 샘플링
  → "이 3D voxel에 해당하는 2D event"를 geometry-aware하게 추출
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from .voxformer_head import VoxFormerHead


class GatedFusion3D(nn.Module):
    """
    seed_feats_rgb (N_q, C) + seed_feats_ev (N_q, C) → fused (N_q, C)

    query별로 RGB/Event 비중을 동적으로 결정:
      동적 물체 query: event 신뢰 (motion signal 강함)
      정적 배경 query: RGB 신뢰 (depth signal 강함)
    """
    def __init__(self, embed_dims: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(embed_dims * 2, embed_dims)

    def forward(self, feat_rgb: torch.Tensor,
                feat_ev: torch.Tensor) -> torch.Tensor:
        """
        feat_rgb: (N_q, C)
        feat_ev:  (N_q, C)
        returns:  (N_q, C)
        """
        gate  = self.gate(torch.cat([feat_rgb, feat_ev], dim=-1))  # (N_q, 1)
        fused = gate * feat_rgb + (1 - gate) * feat_ev
        return fused


@HEADS.register_module()
class VoxFormerHeadEvent(VoxFormerHead):
    """
    3D geometry-aware event cross-attention.

    VoxFormerHead의 forward()를 override하여
    동일한 get_vox_features를 event features에도 적용.

    추가 파라미터:
      use_event_3d: True면 event 3D sampling 활성화
    """

    def __init__(self, *args, use_event_3d: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_event_3d = use_event_3d
        if use_event_3d:
            self.gated_fusion_3d = GatedFusion3D(embed_dims=self.embed_dims)
            print(f"[VoxFormerHeadEvent] 3D geometry-aware event sampling enabled")

    def forward(self, mlvl_feats, img_metas, target,
                ev_feats=None):
        """
        mlvl_feats: list of (B, N, C, H', W')  ← RGB features
        ev_feats:   (B, C, H', W')  ← Event features (optional)
                    None이면 기존 VoxFormerHead와 동일하게 동작
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embed.weight.to(dtype)

        bev_pos_cross = self.positional_encoding(
            torch.zeros((bs, 512, 512),
                        device=bev_queries.device).to(dtype)
        ).to(dtype)
        bev_pos_self = self.positional_encoding(
            torch.zeros((bs, 512, 512),
                        device=bev_queries.device).to(dtype)
        ).to(dtype)

        # ── Proposal 처리 (기존과 동일) ─────────────────────
        proposal = img_metas[0]['proposal'].reshape(
            self.bev_h, self.bev_w, self.bev_z
        )
        unmasked_idx = np.asarray(
            np.where(proposal.reshape(-1) > 0)
        ).astype(np.int32)
        masked_idx = np.asarray(
            np.where(proposal.reshape(-1) == 0)
        ).astype(np.int32)
        vox_coords, ref_3d = self.get_ref_3d()

        # ── RGB Cross-attention (기존과 동일) ─────────────────
        seed_feats_rgb = self.cross_transformer.get_vox_features(
            mlvl_feats, bev_queries,
            self.bev_h, self.bev_w,
            ref_3d=ref_3d, vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos_cross,
            img_metas=img_metas, prev_bev=None,
        )  # list [(N_q, C)] or [(N_q, 1, C)]

        # ── 3D Event Cross-attention (핵심 추가) ──────────────
        if (self.use_event_3d
                and ev_feats is not None
                and hasattr(self, 'gated_fusion_3d')):

            # ev_feats: (B, C, H', W') → mlvl_feats와 동일한 포맷으로 변환
            # (B, N_cam, C, H', W') where 모든 카메라가 동일한 event map을 참조
            # → 각 카메라의 projection matrix로 event map의 해당 위치 샘플링
            B, C, Hf, Wf = ev_feats.shape
            # ev_feats를 FPN output과 동일한 해상도로 맞추기
            if (Hf, Wf) != mlvl_feats[0].shape[-2:]:
                ev_feats_r = F.interpolate(
                    ev_feats, size=mlvl_feats[0].shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                ev_feats_r = ev_feats

            # N_cam 차원으로 확장 (동일한 event map을 N번 복사)
            ev_mlvl = [ev_feats_r.unsqueeze(1).expand(
                B, num_cam, C, *ev_feats_r.shape[-2:]
            )]  # list of [(B, N, C, H', W')]

            # 동일한 get_vox_features로 event sampling
            # → 동일한 3D→2D 투영 행렬 사용 = geometry-consistent!
            seed_feats_ev = self.cross_transformer.get_vox_features(
                ev_mlvl, bev_queries,
                self.bev_h, self.bev_w,
                ref_3d=ref_3d, vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos_cross,
                img_metas=img_metas, prev_bev=None,
            )  # list [(N_q, C)] or [(N_q, 1, C)]

            # shape 안전 처리
            feat_rgb = seed_feats_rgb[0]
            feat_ev  = seed_feats_ev[0]
            if feat_rgb.dim() == 3: feat_rgb = feat_rgb.squeeze(1)
            if feat_ev.dim()  == 3: feat_ev  = feat_ev.squeeze(1)

            # ── GatedFusion3D: 3D 공간에서 RGB + Event 합산 ──
            seed_fused = self.gated_fusion_3d(feat_rgb, feat_ev)
            seed_feats = [seed_fused]

        else:
            # ev_feats 없으면 기존 VoxFormerHead와 동일
            feat_rgb = seed_feats_rgb[0]
            if feat_rgb.dim() == 3: feat_rgb = feat_rgb.squeeze(1)
            seed_feats = [feat_rgb]

        # ── Dense voxel 복원 (기존과 동일) ───────────────────
        vox_feats = torch.empty(
            (self.bev_h, self.bev_w, self.bev_z, self.embed_dims),
            device=bev_queries.device
        )
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = \
            self.mask_embed.weight.view(1, self.embed_dims).expand(
                masked_idx.shape[1], self.embed_dims
            ).to(dtype)

        # ── Self-attention diffusion (기존과 동일) ─────────────
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats, vox_feats_flatten, 512, 512,
            ref_3d=ref_3d, vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos_self,
            img_metas=img_metas, prev_bev=None,
        )
        vox_feats_diff = vox_feats_diff.reshape(
            self.bev_h, self.bev_w, self.bev_z, self.embed_dims
        )
        out = self.header(
            {"x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)}
        )
        return out