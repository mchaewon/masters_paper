"""
VoxFormerHeadEventV2: F_ego/F_obj 분리 3D Cross-Attention
==========================================================
기존 VoxFormerHeadEvent와의 차이:
  V1: ev_feats = conv(concat(F_ego, F_obj)) → 2번 cross-attn (RGB, ev)
      문제: F_ego + F_obj 합산 시 bicyclist 등 rare class에서 interference

  V2: F_ego, F_obj 분리 → 3번 cross-attn (RGB, F_ego, F_obj 각각)
      효과: 각 signal이 독립적으로 3D query에 반영됨

Fusion:
  3-way Gated Fusion:
    gate = softmax(Linear([seed_rgb, seed_ego, seed_obj], 3C → 3))
    seed_fused = gate[0]*seed_rgb + gate[1]*seed_ego + gate[2]*seed_obj

  물리적 의미:
    정적 scene query: gate[1](ego) 높음  → depth 신호 활용
    동적 물체 query:  gate[2](obj) 높음  → velocity 신호 활용
    ambiguous query:  gate[0](rgb) 높음  → RGB feature 신뢰
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from .voxformer_head import VoxFormerHead


class ThreeWayGatedFusion(nn.Module):
    """
    RGB / E_ego / E_obj 세 seed feature를 query별로 adaptive하게 합성.

    gate = softmax(Linear(3C → 3)):
      gate[:,0]: RGB 가중치   (ambiguous/static-semantic)
      gate[:,1]: E_ego 가중치 (depth-rich static region)
      gate[:,2]: E_obj 가중치 (motion-rich dynamic region)
    """
    def __init__(self, embed_dims: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3),
        )  # → (N_q, 3) logits → softmax

    def forward(self,
                seed_rgb: torch.Tensor,
                seed_ego: torch.Tensor,
                seed_obj: torch.Tensor) -> torch.Tensor:
        """
        All inputs: (N_q, C)
        returns:    (N_q, C)
        """
        # query별 3개 가중치 계산
        logits = self.gate(
            torch.cat([seed_rgb, seed_ego, seed_obj], dim=-1)
        )  # (N_q, 3)
        w = torch.softmax(logits, dim=-1)  # (N_q, 3)

        fused = (w[:, 0:1] * seed_rgb +
                 w[:, 1:2] * seed_ego +
                 w[:, 2:3] * seed_obj)    # (N_q, C)
        return fused


@HEADS.register_module()
class VoxFormerHeadEventV2(VoxFormerHead):
    """
    F_ego / F_obj 분리 3D cross-attention + 3-way gated fusion.

    forward 추가 인자:
      ego_feats: (B, C, H', W')  E_ego feature (SUM)
      obj_feats: (B, C, H', W')  E_obj feature (DIFF)
      (기존 ev_feats 인자 대신 두 개로 분리)
    """

    def __init__(self, *args, use_event_3d: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_event_3d = use_event_3d
        if use_event_3d:
            self.three_way_fusion = ThreeWayGatedFusion(self.embed_dims)
            print("[VoxFormerHeadEventV2] 3-way gated fusion (RGB/Ego/Obj) enabled")

    def _feat_to_mlvl(self, feat_2d, num_cam, target_shape):
        """
        (B, C, H', W') → list of [(B, N, C, H', W')]
        동일한 feature를 N개 camera view로 expand
        """
        B, C, Hf, Wf = feat_2d.shape
        if (Hf, Wf) != target_shape:
            feat_2d = F.interpolate(feat_2d, size=target_shape,
                                     mode='bilinear', align_corners=False)
        return [feat_2d.unsqueeze(1).expand(B, num_cam, C, *target_shape)]

    def forward(self, mlvl_feats, img_metas, target,
                ego_feats=None, obj_feats=None,
                ev_feats=None):  # ev_feats: V1 호환용 (무시)
        """
        mlvl_feats: list of (B, N, C, H', W')  ← RGB
        ego_feats:  (B, C, H', W')  ← E_ego (SUM)
        obj_feats:  (B, C, H', W')  ← E_obj (DIFF)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype  = mlvl_feats[0].dtype
        tshape = mlvl_feats[0].shape[-2:]  # (H', W')

        bev_queries = self.bev_embed.weight.to(dtype)
        bev_pos_cross = self.positional_encoding(
            torch.zeros((bs, 512, 512),
                        device=bev_queries.device).to(dtype)
        ).to(dtype)
        bev_pos_self = self.positional_encoding(
            torch.zeros((bs, 512, 512),
                        device=bev_queries.device).to(dtype)
        ).to(dtype)

        # ── Proposal 처리 ──────────────────────────────────────
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

        kwargs = dict(
            bev_h=self.bev_h, bev_w=self.bev_w,
            ref_3d=ref_3d, vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos_cross,
            img_metas=img_metas, prev_bev=None,
        )

        # ── 1. RGB cross-attention ─────────────────────────────
        seed_rgb_raw = self.cross_transformer.get_vox_features(
            mlvl_feats, bev_queries, **kwargs
        )
        seed_rgb = seed_rgb_raw[0]
        if seed_rgb.dim() == 3:
            seed_rgb = seed_rgb.squeeze(1)  # (N_q, C)

        # ── 2. E_ego cross-attention (분리!) ───────────────────
        use_3d = (self.use_event_3d
                  and ego_feats is not None
                  and obj_feats is not None
                  and hasattr(self, 'three_way_fusion'))

        if use_3d:
            # E_ego: depth signal → 정적 scene query에 유리
            ego_mlvl = self._feat_to_mlvl(ego_feats, num_cam, tshape)
            seed_ego_raw = self.cross_transformer.get_vox_features(
                ego_mlvl, bev_queries, **kwargs
            )
            seed_ego = seed_ego_raw[0]
            if seed_ego.dim() == 3:
                seed_ego = seed_ego.squeeze(1)  # (N_q, C)

            # ── 3. E_obj cross-attention (분리!) ──────────────
            # E_obj: motion signal → 동적 물체 query에 유리
            obj_mlvl = self._feat_to_mlvl(obj_feats, num_cam, tshape)
            seed_obj_raw = self.cross_transformer.get_vox_features(
                obj_mlvl, bev_queries, **kwargs
            )
            seed_obj = seed_obj_raw[0]
            if seed_obj.dim() == 3:
                seed_obj = seed_obj.squeeze(1)  # (N_q, C)

            # ── 4. 3-way Gated Fusion ─────────────────────────
            seed_final = self.three_way_fusion(seed_rgb, seed_ego, seed_obj)

        else:
            seed_final = seed_rgb

        # ── Dense voxel 복원 ───────────────────────────────────
        vox_feats = torch.empty(
            (self.bev_h, self.bev_w, self.bev_z, self.embed_dims),
            device=bev_queries.device
        )
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_final
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = \
            self.mask_embed.weight.view(1, self.embed_dims).expand(
                masked_idx.shape[1], self.embed_dims
            ).to(dtype)

        # ── Self-attention diffusion ────────────────────────────
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
        return self.header(
            {"x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)}
        )