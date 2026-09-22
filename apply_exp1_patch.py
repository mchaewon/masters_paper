"""
실험 1 패치: RGB + Event Voxel Grid Concat Input
=================================================
수정 내용:
  1. semantic_kitti_dataset_stage2.py
     get_input_info(): RGB(3ch) + Event(3ch) → [N, 6, H, W]

  2. voxformer.py (detectors)
     extract_img_feat(): 6채널 입력 처리
     pretrained weight 로딩 시 event 채널 zero-init

  3. VoxFormer-T.py (config)
     in_channels: 3 → 6 (FPN 입력 채널)

실행:
    # 원본 백업
    cp .../semantic_kitti_dataset_stage2.py ...stage2_exp0.py
    cp .../voxformer.py .../voxformer_exp0.py

    python apply_exp1_patch.py
"""

import os, shutil

DATASET_FILE = "/VoxFormer/projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_stage2.py"
DETECTOR_FILE = "/VoxFormer/projects/mmdet3d_plugin/voxformer/detectors/voxformer.py"


# ============================================================
# 패치 1: dataset - get_input_info() 수정
# RGB + Event를 채널 방향으로 concat하여 6채널 반환
# ============================================================
def patch_dataset():
    with open(DATASET_FILE) as f:
        src = f.read()

    # 실험 0 버전의 get_input_info (event only) → 실험 1 (RGB + event)로 교체
    old = '''\
        """Get the image of the specific frame in a sequence.
        [Exp0] RGB 대신 Event Voxel Grid를 로드한다.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Event input [N, 3, img_H, img_W]
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        # 현재 프레임 event
        image_list.append(self.load_event_as_rgb(sequence, frame_id))

        # temporal 프레임 event
        for i in self.target_frames:
            id = int(frame_id)
            if id + i < 0 or id + i > seq_len - 1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)
            image_list.append(self.load_event_as_rgb(sequence, target_id))

        image_tensor = torch.stack(image_list, dim=0)  # [5, 3, 370, 1220]
        return image_tensor'''

    new = '''\
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

    if old in src:
        src = src.replace(old, new, 1)
        with open(DATASET_FILE, 'w') as f:
            f.write(src)
        print("✅ 패치 1 완료: get_input_info → RGB+Event concat [N,6,H,W]")
    else:
        print("⚠️  패치 1: Exp0 버전 못 찾음. 원본(RGB) 버전에서 시작하는지 확인")
        # 원본 RGB 버전에서 시작하는 경우
        old_orig = '''\
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        image_list.append(self.normalize_rgb(img))

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]  # crop image

            image_list.append(self.normalize_rgb(img))

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]

        return image_tensor'''

        if old_orig in src:
            # new_exp1을 여기에도 적용
            new_from_orig = new  # 동일한 결과
            src = src.replace(old_orig, new_from_orig, 1)
            # event 설정도 없으면 추가
            if "self.event_root" not in src:
                old_hw = "        self.img_W = 1220\n        self.img_H = 370"
                new_hw = """\
        self.img_W = 1220
        self.img_H = 370

        # Exp1: Event 설정
        self.event_root = os.path.join(data_root, "events", "sequences")
        self.num_event_bins = 5
        self.ev_H, self.ev_W = 352, 1216"""
                src = src.replace(old_hw, new_hw, 1)
            with open(DATASET_FILE, 'w') as f:
                f.write(src)
            print("✅ 패치 1 완료 (원본 버전 기준)")
        else:
            print("❌ 패치 1 실패: 수동 확인 필요")


# ============================================================
# 패치 2: voxformer.py - 6채널 입력 + pretrained weight 처리
# ============================================================
def patch_detector():
    with open(DETECTOR_FILE) as f:
        src = f.read()

    # extract_img_feat에서 6채널 처리 (기존 코드와 호환됨, 변경 불필요)
    # 핵심은 pretrained weight 로딩 시 첫 conv를 3→6으로 확장하는 것
    # __init__ 마지막에 weight 확장 코드 추가

    old = '''\
        super(VoxFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)'''

    new = '''\
        super(VoxFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        # ── Exp1: 첫 conv를 3ch → 6ch로 확장 ─────────────────
        # RGB pretrained weight 유지, event 채널은 zero-init
        self._expand_first_conv_to_6ch()
        # ──────────────────────────────────────────────────────'''

    expand_method = '''
    def _expand_first_conv_to_6ch(self):
        """
        ResNet 첫 번째 conv layer를 3ch → 6ch로 확장한다.
        - RGB 3채널: pretrained weight 그대로 유지
        - Event 3채널: zero-init (학습 초기에 RGB 위주로 동작)
        """
        import torch.nn as nn
        first_conv = self.img_backbone.layer0[0] \
            if hasattr(self.img_backbone, 'layer0') \
            else self.img_backbone.conv1

        if first_conv.in_channels == 6:
            return  # 이미 확장됨

        old_weight = first_conv.weight.data.clone()  # [64, 3, 7, 7]
        new_conv = nn.Conv2d(
            6, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        # RGB 채널: pretrained weight 복사
        new_conv.weight.data[:, :3, :, :] = old_weight
        # Event 채널: zero-init
        new_conv.weight.data[:, 3:, :, :] = 0.0
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

        # 교체
        if hasattr(self.img_backbone, 'layer0'):
            self.img_backbone.layer0[0] = new_conv
        else:
            self.img_backbone.conv1 = new_conv

        print(f"[Exp1] First conv expanded: 3ch -> 6ch "
              f"(RGB pretrained, Event zero-init)")

'''

    if old in src:
        src = src.replace(old, new, 1)
        # expand_method를 extract_img_feat 앞에 삽입
        insert_before = "    def extract_img_feat"
        src = src.replace(insert_before,
                          expand_method + "    def extract_img_feat", 1)
        with open(DETECTOR_FILE, 'w') as f:
            f.write(src)
        print("✅ 패치 2 완료: 6ch first conv 확장 메서드 추가")
    else:
        print("❌ 패치 2 실패: __init__ super() 구문 못 찾음")


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("="*55)
    print("실험 1 패치: RGB + Event Voxel Grid Concat")
    print("="*55)

    # 백업
    for f, tag in [(DATASET_FILE, "exp0"), (DETECTOR_FILE, "exp0")]:
        backup = f.replace(".py", f"_{tag}.py")
        if not os.path.exists(backup):
            shutil.copy2(f, backup)
            print(f"[백업] {backup}")

    patch_dataset()
    patch_detector()

    print("\n" + "="*55)
    print("다음 단계:")
    print("  1. config에서 FPN in_channels 수정 (아래 참고)")
    print("  2. python tools/train.py 또는 test.py 실행")
    print("="*55)
    print("""
[Config 수정 필요 - VoxFormer-T.py]
img_neck=dict(
    type='FPN',
    in_channels=[1024],   # ← 변경 없음 (FPN은 backbone output 채널)
    ...
)
# backbone output은 여전히 1024ch이므로 FPN은 수정 불필요
# 단, backbone의 첫 conv만 6ch로 바뀜
""")