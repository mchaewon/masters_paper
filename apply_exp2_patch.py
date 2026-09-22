"""
실험 2 패치: Dedicated Event Encoder + Feature-level Fusion
============================================================
구조:
  RGB  (3,H,W) → ResNet backbone → FPN → RGB feat (C,H',W')  ─┐
                                                                  ├─ fusion → Lifting → 3D
  Event(B,H,W) → EventEncoder  → event feat (C,H',W') ─────────┘

Exp 1과의 차이:
  Exp 1: 공유 backbone이 [RGB;Event] 6ch 처리 (early fusion)
  Exp 2: 전용 encoder가 event 별도 처리 후 feature 수준에서 fusion (late fusion)

수정 파일:
  1. semantic_kitti_dataset_stage2.py: RGB 복원 + event를 img_metas에 추가
  2. voxformer.py: EventEncoder + fusion module 추가
  3. config: in_channels 복원 (3ch)

실행:
    python apply_exp2_patch.py
"""

import os, shutil

DATASET_FILE = "/VoxFormer/projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_stage2.py"
DETECTOR_FILE = "/VoxFormer/projects/mmdet3d_plugin/voxformer/detectors/voxformer.py"
CONFIG_FILE   = "/VoxFormer/projects/configs/voxformer/voxformer-T_deform3D.py"


# ============================================================
# 패치 1: dataset - get_input_info() RGB 복원
#                  event는 img_metas에 경로만 추가
# ============================================================
def patch_dataset():
    with open(DATASET_FILE) as f:
        src = f.read()

    # get_input_info를 RGB 원본으로 복원
    old = '''\
        """Get the image of the specific frame in a sequence.
        [Exp1] RGB(3ch) + Event(3ch) concat -> [N, 6, H, W]

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Fused input [N, 6, img_H, img_W]
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        # ── 현재 프레임: RGB + Event concat ──────────────────
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences",
            sequence, "image_2", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop (370, 1220)
        rgb_tensor = self.normalize_rgb(img)              # [3, H, W]
        ev_tensor  = self.load_event_as_rgb(sequence, frame_id)  # [3, H, W]
        fused = torch.cat([rgb_tensor, ev_tensor], dim=0) # [6, H, W]
        image_list.append(fused)

        # ── temporal 프레임: RGB + Event concat ──────────────
        for i in self.target_frames:
            id = int(frame_id)
            if id + i < 0 or id + i > seq_len - 1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences",
                sequence, "image_2", target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]
            rgb_t = self.normalize_rgb(img)
            ev_t   = self.load_event_as_rgb(sequence, target_id)
            fused_t = torch.cat([rgb_t, ev_t], dim=0)  # [6, H, W]
            image_list.append(fused_t)

        image_tensor = torch.stack(image_list, dim=0)  # [5, 6, 370, 1220]
        return image_tensor'''

    new = '''\
        """Get the image of the specific frame in a sequence.
        [Exp2] RGB only. Event는 별도 encoder로 처리.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: RGB input [N, 3, img_H, img_W]
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        # 현재 프레임 RGB
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences",
            sequence, "image_2", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]
        image_list.append(self.normalize_rgb(img))

        # temporal 프레임 RGB
        for i in self.target_frames:
            id = int(frame_id)
            if id + i < 0 or id + i > seq_len - 1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)
            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences",
                sequence, "image_2", target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]
            image_list.append(self.normalize_rgb(img))

        image_tensor = torch.stack(image_list, dim=0)  # [5, 3, 370, 1220]
        return image_tensor'''

    if old in src:
        src = src.replace(old, new, 1)
        print("✅ 패치 1a: get_input_info RGB 복원")
    else:
        print("⚠️  패치 1a: Exp1 버전 못 찾음 (이미 RGB거나 다른 상태)")

    # get_meta_info에 event_path 추가
    old_meta = '''        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            proposal=proposal_bin,
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            img_shape = [(self.img_H,self.img_W)]
        )'''

    new_meta = '''        # Exp2: event 경로를 meta에 포함
        event_path = os.path.join(
            self.event_root, sequence, f"{frame_id}.npy"
        )

        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            proposal=proposal_bin,
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            img_shape = [(self.img_H,self.img_W)],
            event_path = event_path,   # Exp2 추가
        )'''

    if old_meta in src:
        src = src.replace(old_meta, new_meta, 1)
        print("✅ 패치 1b: event_path를 img_metas에 추가")
    else:
        print("⚠️  패치 1b: meta_dict 패턴 못 찾음")

    with open(DATASET_FILE, 'w') as f:
        f.write(src)


# ============================================================
# 패치 2: voxformer.py - EventEncoder + fusion 추가
# ============================================================
def patch_detector():
    with open(DETECTOR_FILE) as f:
        src = f.read()

    # import 추가
    old_import = "import time\nimport copy\nimport torch\nimport numpy as np"
    new_import = "import time\nimport copy\nimport torch\nimport torch.nn as nn\nimport numpy as np"
    if old_import in src:
        src = src.replace(old_import, new_import, 1)
        print("✅ 패치 2a: nn import 추가")

    # __init__에 EventEncoder 초기화 추가
    # Exp1의 _expand_first_conv_to_6ch 제거하고 EventEncoder로 교체
    old_init_exp1 = '''\
        # ── Exp1: 첫 conv를 3ch → 6ch로 확장 ─────────────────
        # RGB pretrained weight 유지, event 채널은 zero-init
        self._expand_first_conv_to_6ch()
        # ──────────────────────────────────────────────────────'''

    new_init_exp2 = '''\
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
        # ── Exp2 End ──────────────────────────────────────────'''

    if old_init_exp1 in src:
        src = src.replace(old_init_exp1, new_init_exp2, 1)
        print("✅ 패치 2b: EventEncoder 초기화 (Exp1 코드 교체)")
    else:
        # Exp1 코드 없으면 super().__init__ 바로 뒤에 삽입
        old_super = '''        super(VoxFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)'''
        new_super = old_super + '\n\n' + new_init_exp2
        if old_super in src:
            src = src.replace(old_super, new_super, 1)
            print("✅ 패치 2b: EventEncoder 초기화 삽입")

    # Exp1의 _expand_first_conv_to_6ch 메서드 제거 (있으면)
    if '_expand_first_conv_to_6ch' in src:
        # 메서드 전체 제거
        start = src.find('    def _expand_first_conv_to_6ch(self):')
        if start >= 0:
            # 다음 def 찾기
            next_def = src.find('\n    def ', start + 1)
            if next_def >= 0:
                src = src[:start] + src[next_def + 1:]
                print("✅ 패치 2c: Exp1 메서드 제거")

    # extract_event_feat 메서드 추가
    event_feat_method = '''
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

'''

    # extract_img_feat 바로 앞에 삽입
    insert_before = "    def extract_img_feat"
    if insert_before in src and 'extract_event_feat' not in src:
        src = src.replace(insert_before,
                          event_feat_method + "    def extract_img_feat", 1)
        print("✅ 패치 2d: extract_event_feat 메서드 추가")

    # forward_train 수정: event 로딩 + fusion 추가
    old_train = '''\
        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img) 
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses'''

    new_train = '''\
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
        return losses'''

    if old_train in src:
        src = src.replace(old_train, new_train, 1)
        print("✅ 패치 2e: forward_train에 event fusion 추가")

    # forward_test 수정
    old_test = '''\
        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img) 
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        # dict이면 list로 감싸서 single_gpu_test의 extend와 호환
        if isinstance(completion_results, dict):
            completion_results = [completion_results]

        return completion_results'''

    new_test = '''\
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

        return completion_results'''

    if old_test in src:
        src = src.replace(old_test, new_test, 1)
        print("✅ 패치 2f: forward_test에 event fusion 추가")

    # _fuse_event_feats 메서드 추가
    fuse_method = '''
    def _fuse_event_feats(self, img_feats, img_metas, img_H, img_W):
        """
        Event voxel을 로드하여 RGB feature와 fusion.
        img_feats: list of [B, N_cam, C, H', W']
        """
        import numpy as np
        import torch.nn.functional as F

        # img_metas에서 event_path 가져오기
        event_paths = [m.get('event_path', None) for m in img_metas]
        if all(p is None for p in event_paths):
            return img_feats  # event 없으면 그냥 통과

        B = len(event_paths)
        T = 5   # num_event_bins

        # event voxel 로드: (B, T, ev_H, ev_W)
        ev_H, ev_W = 352, 1216
        batch_voxels = []
        for path in event_paths:
            if path and os.path.exists(path):
                events = np.load(path)
                voxel = np.zeros((T, ev_H, ev_W), dtype=np.float32)
                if events.ndim == 2 and events.shape[1] == 4:
                    x = events[:, 0].astype(np.int32)
                    y = events[:, 1].astype(np.int32)
                    t = events[:, 2].astype(np.float64)
                    p = events[:, 3].astype(np.float32)
                    ok = (x >= 0) & (x < ev_W) & (y >= 0) & (y < ev_H)
                    x, y, t, p = x[ok], y[ok], t[ok], p[ok]
                    if len(t) > 0:
                        t_min, t_max = t.min(), t.max()
                        tn = (t - t_min) / (t_max - t_min + 1e-8) * (T - 1)
                        pol = np.where(p > 0, 1.0, -1.0).astype(np.float32)
                        tf = tn.astype(np.int32)
                        tc = tf + 1
                        wc = (tn - tf).astype(np.float32)
                        wf = 1.0 - wc
                        mf = tf < T
                        if mf.sum():
                            np.add.at(voxel, (tf[mf], y[mf], x[mf]),
                                      pol[mf] * wf[mf])
                        mc = tc < T
                        if mc.sum():
                            np.add.at(voxel, (tc[mc], y[mc], x[mc]),
                                      pol[mc] * wc[mc])
            else:
                voxel = np.zeros((T, ev_H, ev_W), dtype=np.float32)
            batch_voxels.append(voxel)

        ev_tensor = torch.from_numpy(
            np.stack(batch_voxels, axis=0)
        ).float().to(img_feats[0].device)  # (B, T, ev_H, ev_W)

        # img_H, img_W로 resize
        ev_tensor = F.interpolate(
            ev_tensor, size=(img_H, img_W), mode='bilinear', align_corners=False
        )  # (B, T, img_H, img_W)

        # EventEncoder: (B, T, H, W) → (B, C, H', W')
        ev_feat = self.event_encoder(ev_tensor)  # (B, 128, H', W')

        # img_feats와 fusion
        # img_feats[0]: (B, N_cam, C, H', W')
        fused_feats = []
        for feat in img_feats:
            B_f, N_cam, C, Hf, Wf = feat.shape
            # ev_feat을 feat 해상도에 맞게 resize
            ev_resized = F.interpolate(
                ev_feat, size=(Hf, Wf), mode='bilinear', align_corners=False
            )  # (B, C, Hf, Wf)
            # N_cam 차원으로 확장
            ev_expanded = ev_resized.unsqueeze(1).expand(B_f, N_cam, C, Hf, Wf)
            # concat → fusion_conv
            fused = torch.cat([feat, ev_expanded], dim=2)  # (B, N, 2C, H', W')
            # reshape for conv
            fused = fused.view(B_f * N_cam, 2 * C, Hf, Wf)
            fused = self.fusion_conv(fused)                 # (B*N, C, H', W')
            fused = fused.view(B_f, N_cam, C, Hf, Wf)
            fused_feats.append(fused)

        return fused_feats

'''

    # extract_img_feat 바로 앞에 삽입
    if '_fuse_event_feats' not in src:
        src = src.replace(
            "    def extract_img_feat",
            fuse_method + "    def extract_img_feat", 1
        )
        print("✅ 패치 2g: _fuse_event_feats 메서드 추가")

    with open(DETECTOR_FILE, 'w') as f:
        f.write(src)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("실험 2 패치: Dedicated Event Encoder + Feature Fusion")
    print("=" * 55)

    # 백업
    for f in [DATASET_FILE, DETECTOR_FILE]:
        backup = f.replace(".py", "_exp1.py")
        if not os.path.exists(backup):
            shutil.copy2(f, backup)
            print(f"[백업] {backup}")

    patch_dataset()
    patch_detector()

    print("""
✅ 패치 완료

학습 실행:
  python tools/train.py \\
      ./projects/configs/voxformer/voxformer-T_deform3D.py \\
      --cfg-options load_from=ckpts/voxformer-T-3D/miou13.69_iou44.34_epoch_15.pth \\
      --work-dir results/exp2_event_encoder

※ config는 수정 불필요
  - FPN in_channels=[1024]: backbone output 채널 (변경 없음)
  - img_backbone: 3채널 입력 복원됨
  - EventEncoder와 FusionConv는 scratch 학습됨
    (RGB backbone은 pretrained weight 유지)
""")