"""
VoxFormerHeadBC: VoxFormerHead + Method B + Method C
=====================================================
Method C: Event-derived query augmentation
  F_obj high-activation → 3D backproject → proposal augmentation
  → get_vox_features가 dynamic object 위치도 처리

Method B: Dual-stream 3D lifting
  RGB stream:   get_vox_features(mlvl_feats)  → seed_feats_rgb
  Event stream: event_cross_attn(event_feat)  → seed_feats_ev
  GatedFusion:  seed_feats_rgb + seed_feats_ev → seed_feats_fused

기존 VoxFormerHead.forward()의 흐름을 유지하면서
proposal 생성과 seed_feats 계산 사이에 두 방법을 삽입.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from .voxformer_head import VoxFormerHead   # 기존 head 상속


# ============================================================
# Method B: Event Cross-Attention Stream
# ============================================================
class EventCrossAttn(nn.Module):
    """
    proposal query가 event 2D feature에 attend하는 경량 cross-attention.
    get_vox_features와 달리 카메라 projection 없이
    feature map 전체를 key/value로 사용.

    입력:
      queries:   (N_q, C) occupied voxel의 bev_query embeddings
      event_2d:  (B, C, H', W') event feature map
    출력:
      (N_q, C) event-attended features
    """
    def __init__(self, embed_dims=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dims, num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dims * 2, embed_dims),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_layers)
        ])

    def forward(self, queries, event_2d, chunk_size=2048):
        """
        queries:    (N_q, C)
        event_2d:   (B, C, H', W')
        chunk_size: 메모리 절감을 위한 query chunk 크기
        returns:    (N_q, C)
        """
        B, C, H, W = event_2d.shape
        N_q = queries.shape[0]

        # event feature를 추가로 downsample (메모리 절감)
        # (B, C, H', W') → (B, C, H'/2, W'/2)
        ev_small = F.avg_pool2d(event_2d, kernel_size=2, stride=2)
        ev = ev_small.flatten(2).permute(0, 2, 1)  # (B, H*W/4, C)

        # Chunked attention: N_q를 chunk_size씩 나눠서 처리
        outputs = []
        for start in range(0, N_q, chunk_size):
            end = min(start + chunk_size, N_q)
            q_chunk = queries[start:end].unsqueeze(0).expand(B, -1, -1)

            for attn, norm, ffn, ffn_norm in zip(
                self.layers, self.norms, self.ffns, self.ffn_norms
            ):
                q_norm = norm(q_chunk)
                attn_out, _ = attn(
                    query=q_norm,
                    key=ev,
                    value=ev
                )
                q_chunk = q_chunk + attn_out
                q_chunk = q_chunk + ffn(ffn_norm(q_chunk))

            outputs.append(q_chunk.squeeze(0))  # (chunk, C)

        return torch.cat(outputs, dim=0)  # (N_q, C)


# ============================================================
# Gated 3D Fusion
# ============================================================
class GatedFusion3D(nn.Module):
    """
    RGB seed features + Event seed features → fused
    gate는 query별로 RGB/Event 비중을 결정.
    static query: gate→1 (RGB 위주)
    dynamic query: gate→0 (Event 위주)
    """
    def __init__(self, embed_dims=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(embed_dims * 2, embed_dims)

    def forward(self, feat_rgb, feat_ev):
        """
        feat_rgb: (N_q, C)
        feat_ev:  (N_q, C)
        returns:  (N_q, C)
        """
        gate = self.gate(
            torch.cat([feat_rgb, feat_ev], dim=-1)
        )  # (N_q, 1)

        # gate=1: RGB 신뢰 (static), gate=0: Event 신뢰 (dynamic)
        fused = gate * feat_rgb + (1 - gate) * feat_ev
        return fused


# ============================================================
# Method C: Event-derived Query Generator
# ============================================================
def generate_event_proposals(F_obj, depth_map, img_metas,
                              bev_h=128, bev_w=128, bev_z=16,
                              vox_origin=None, vox_size=0.2,
                              top_k=300):
    """
    F_obj의 high-activation 2D 위치를 3D backproject하여
    binary proposal mask에 추가할 dynamic positions를 생성.

    F_obj:       (B, C, Hf, Wf)  obj event feature
    depth_map:   (B, 1, H, W)    depth in meters
    img_metas:   list of meta dicts (cam_intrinsic, lidar2cam 포함)
    bev_h/w/z:   voxel grid 크기 (VoxFormer 기본: 128×128×16)
    vox_origin:  [0, -25.6, -2] (VoxFormer 기본)
    vox_size:    0.2 (m)
    top_k:       추가할 dynamic query 최대 수

    returns:
      event_proposal: (bev_h, bev_w, bev_z) binary numpy array
    """
    if vox_origin is None:
        vox_origin = [0, -25.6, -2]

    B, C, Hf, Wf = F_obj.shape
    H = depth_map.shape[2]
    W = depth_map.shape[3]

    event_proposal = np.zeros((bev_h, bev_w, bev_z), dtype=np.float32)

    try:
        # 1. F_obj activation map → top-K 2D 위치
        activation = F_obj[0].detach().mean(dim=0)  # (Hf, Wf)
        topk_vals, topk_flat = activation.flatten().topk(
            min(top_k, Hf * Wf)
        )

        # activation threshold: 상위 50% 이상만
        threshold = topk_vals[-1] * 0.5
        valid_topk = topk_flat[topk_vals >= threshold]
        if len(valid_topk) == 0:
            return event_proposal

        u_feat = (valid_topk % Wf).float()   # (K,)
        v_feat = (valid_topk // Wf).float()  # (K,)

        # 2. feature 해상도 → 이미지 해상도 스케일
        scale_w = W / Wf
        scale_h = H / Hf
        u_img = (u_feat * scale_w).long().clamp(0, W - 1)
        v_img = (v_feat * scale_h).long().clamp(0, H - 1)

        # 3. depth 값 추출
        depths = depth_map[0, 0, v_img, u_img]  # (K,)
        depth_valid = depths > 0.5
        if depth_valid.sum() == 0:
            return event_proposal

        u_v = u_img[depth_valid].float()
        v_v = v_img[depth_valid].float()
        d_v = depths[depth_valid]

        # 4. Camera intrinsics
        cam_K = img_metas[0]['cam_intrinsic'][0]  # (3, 3)
        fx = cam_K[0, 0]; fy = cam_K[1, 1]
        cx = cam_K[0, 2]; cy = cam_K[1, 2]

        # 5. 2D → 3D (카메라 좌표계)
        x_cam = (u_v - cx) * d_v / fx
        y_cam = (v_v - cy) * d_v / fy
        z_cam = d_v
        xyz_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)

        # 6. 카메라 → LiDAR 좌표
        lidar2cam = torch.tensor(
            img_metas[0]['lidar2cam'][0], dtype=torch.float32
        ).to(F_obj.device)  # (4, 4)
        T_inv = torch.inverse(lidar2cam)
        xyz_lidar = (T_inv @ xyz_cam.T).T[:, :3]  # (K', 3)

        # 7. LiDAR → Voxel index
        origin = torch.tensor(vox_origin, dtype=torch.float32).to(F_obj.device)
        vox_idx = ((xyz_lidar - origin) / vox_size).long()

        # 8. 범위 내 필터링
        mask = (
            (vox_idx[:, 0] >= 0) & (vox_idx[:, 0] < bev_h) &
            (vox_idx[:, 1] >= 0) & (vox_idx[:, 1] < bev_w) &
            (vox_idx[:, 2] >= 0) & (vox_idx[:, 2] < bev_z)
        )
        vox_idx = vox_idx[mask].cpu().numpy()

        # 9. Binary mask에 추가
        if len(vox_idx) > 0:
            event_proposal[vox_idx[:, 0],
                           vox_idx[:, 1],
                           vox_idx[:, 2]] = 1.0

    except Exception as e:
        # Query 생성 실패 시 빈 proposal 반환 (학습 중단 방지)
        print(f"[WARN] generate_event_proposals failed: {e}")

    return event_proposal


# ============================================================
# VoxFormerHeadBC
# ============================================================
@HEADS.register_module()
class VoxFormerHeadBC(VoxFormerHead):
    """
    VoxFormerHead + Method B (dual-stream) + Method C (event query)

    추가 파라미터:
      use_event_query: Method C 사용 여부
      use_dual_stream: Method B 사용 여부
      event_top_k:     Method C에서 추가할 dynamic query 수
    """

    def __init__(self,
                 *args,
                 use_event_query=True,
                 use_dual_stream=True,
                 event_top_k=300,
                 event_attn_layers=2,
                 event_attn_heads=4,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.use_event_query  = use_event_query
        self.use_dual_stream  = use_dual_stream
        self.event_top_k      = event_top_k

        if use_dual_stream:
            # Method B: Event cross-attention stream
            self.event_cross_attn = EventCrossAttn(
                embed_dims=self.embed_dims,
                num_heads=event_attn_heads,
                num_layers=event_attn_layers,
            )
            # Gated 3D fusion
            self.gated_fusion_3d = GatedFusion3D(
                embed_dims=self.embed_dims
            )
            print(f"[VoxFormerHeadBC] Method B (dual-stream) enabled")

        if use_event_query:
            print(f"[VoxFormerHeadBC] Method C (event query) enabled, top_k={event_top_k}")

    def forward(self, mlvl_feats, img_metas, target,
                event_feat_2d=None, F_obj=None, depth_map=None):
        """
        기존 VoxFormerHead.forward()에 Method B+C를 삽입.

        추가 인자:
          event_feat_2d: (B, C, H', W') E_ego+E_obj concat feature
          F_obj:         (B, C, H', W') object motion feature (Method C용)
          depth_map:     (B, 1, H, W)   depth map (Method C용)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embed.weight.to(dtype)  # (H*W*Z, C)

        # Positional embeddings
        bev_pos_cross_attn = self.positional_encoding(
            torch.zeros((bs, 512, 512),
                        device=bev_queries.device).to(dtype)
        ).to(dtype)
        bev_pos_self_attn = self.positional_encoding(
            torch.zeros((bs, 512, 512),
                        device=bev_queries.device).to(dtype)
        ).to(dtype)

        # ── Method C: Proposal Augmentation ──────────────────
        proposal = img_metas[0]['proposal'].reshape(
            self.bev_h, self.bev_w, self.bev_z
        )

        if (self.use_event_query
                and F_obj is not None
                and depth_map is not None
                and self.training):   # 학습 시에만 augmentation
            with torch.no_grad():    # query 생성은 gradient 불필요
                event_prop = generate_event_proposals(
                    F_obj.detach(), depth_map.detach(), img_metas,
                    bev_h=self.bev_h, bev_w=self.bev_w, bev_z=self.bev_z,
                    vox_origin=[0, -25.6, -2], vox_size=0.2,
                    top_k=self.event_top_k
                )
            proposal_aug = np.clip(
                proposal.astype(np.float32) + event_prop, 0, 1
            )
            n_added = (proposal_aug > proposal).sum()
        else:
            proposal_aug = proposal
            n_added = 0

        unmasked_idx = np.asarray(
            np.where(proposal_aug.reshape(-1) > 0)
        ).astype(np.int32)
        masked_idx = np.asarray(
            np.where(proposal_aug.reshape(-1) == 0)
        ).astype(np.int32)

        vox_coords, ref_3d = self.get_ref_3d()

        # ── RGB Cross-attention (기존) ────────────────────────
        seed_feats_rgb = self.cross_transformer.get_vox_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
        )  # list of [(N_q, C)]

        # ── Method B: Event Cross-attention Stream ────────────
        if (self.use_dual_stream
                and event_feat_2d is not None
                and hasattr(self, 'event_cross_attn')):

            # seed_feats_rgb[0] shape 안전 처리
            # encoder 반환이 (N_q, C) 또는 (N_q, 1, C) 모두 처리
            rgb_feat = seed_feats_rgb[0]
            if rgb_feat.dim() == 3:
                rgb_feat = rgb_feat.squeeze(1)  # (N_q, 1, C) → (N_q, C)

            # occupied voxel의 query embeddings 추출 (원본 bev_queries 사용)
            occupied_queries = bev_queries[
                vox_coords[unmasked_idx[0], 3]
            ]  # (N_q, C)

            # Event cross-attention
            # event_feat_2d: gradient는 detector에서 이미 계산됨
            # head에서는 feature를 읽기만 함 → 메모리 절감
            seed_feats_ev = self.event_cross_attn(
                occupied_queries, event_feat_2d
            )  # (N_q, C)

            # Gated 3D fusion
            seed_feats_fused = self.gated_fusion_3d(
                rgb_feat,      # (N_q, C)
                seed_feats_ev  # (N_q, C)
            )  # (N_q, C)

            seed_feats = [seed_feats_fused]
        else:
            # shape 안전 처리
            rgb_feat = seed_feats_rgb[0]
            if rgb_feat.dim() == 3:
                rgb_feat = rgb_feat.squeeze(1)
            seed_feats = [rgb_feat]

        # ── Dense voxel 복원 (기존과 동일) ───────────────────
        vox_feats = torch.empty(
            (self.bev_h, self.bev_w, self.bev_z, self.embed_dims),
            device=bev_queries.device
        )
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)

        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = \
            seed_feats[0]
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = \
            self.mask_embed.weight.view(1, self.embed_dims).expand(
                masked_idx.shape[1], self.embed_dims
            ).to(dtype)

        # ── Self-attention diffusion (기존과 동일) ─────────────
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            vox_feats_flatten,
            512, 512,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
        )
        vox_feats_diff = vox_feats_diff.reshape(
            self.bev_h, self.bev_w, self.bev_z, self.embed_dims
        )

        input_dict = {
            "x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)
        }
        out = self.header(input_dict)
        return out