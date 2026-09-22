"""
실험 0 패치 자동 적용 스크립트
================================
semantic_kitti_dataset_stage2.py에 event 로딩 코드를 자동으로 삽입한다.

실행:
    python apply_exp0_patch.py

    # 원본 복사본 먼저 만들기 (권장)
    cp /VoxFormer/projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_stage2.py \
       /VoxFormer/projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_stage2_orig.py
"""

import re
import shutil
import os

TARGET = "/VoxFormer/projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_stage2.py"
BACKUP = TARGET.replace(".py", "_orig.py")


def apply_patch():
    # 원본 백업
    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print(f"[백업] {BACKUP}")

    with open(TARGET, 'r') as f:
        src = f.read()

    # ── 패치 1: __init__에 event 설정 추가 ──────────────────
    old_1 = "        self.img_W = 1220\n        self.img_H = 370"
    new_1 = """\
        self.img_W = 1220
        self.img_H = 370

        # ── Exp0: Event 설정 ──────────────────────────────────
        self.event_root = os.path.join(data_root, "events", "sequences")
        self.num_event_bins = 5
        self.ev_H, self.ev_W = 352, 1216
        # ── Exp0 End ──────────────────────────────────────────"""

    assert old_1 in src, "[ERROR] 패치 1 삽입 위치를 찾지 못함. 원본 파일을 확인하세요."
    src = src.replace(old_1, new_1, 1)
    print("[완료] 패치 1: event 설정 추가")

    # ── 패치 2: load_event_as_rgb 메서드 삽입 ───────────────
    # get_input_info 메서드 바로 앞에 삽입
    insert_before = "    def get_input_info(self, sequence, frame_id):"
    new_method = '''\
    def load_event_as_rgb(self, sequence, frame_id):
        """
        Raw events (N,4) -> 3채널 텐서 [3, img_H, img_W]

        변환:
          1. Raw events -> Voxel Grid (B, ev_H, ev_W)
          2. 양/음 극성 분리 -> 3채널
          3. Resize to (img_H, img_W)
          4. ImageNet 정규화
        """
        import cv2

        event_path = os.path.join(
            self.event_root, sequence, f"{frame_id}.npy"
        )

        if not os.path.exists(event_path):
            return torch.zeros(3, self.img_H, self.img_W)

        events = np.load(event_path)  # (N, 4): [x, y, t, p]

        B = self.num_event_bins
        H, W = self.ev_H, self.ev_W
        voxel = np.zeros((B, H, W), dtype=np.float32)

        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2].astype(np.float64)
        p = events[:, 3].astype(np.float32)

        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]

        if len(t) > 0:
            t_min, t_max = t.min(), t.max()
            t_norm = (t - t_min) / (t_max - t_min + 1e-8) * (B - 1)
            pol    = np.where(p > 0, 1.0, -1.0).astype(np.float32)
            t_floor = t_norm.astype(np.int32)
            t_ceil  = t_floor + 1
            w_ceil  = (t_norm - t_floor).astype(np.float32)
            w_floor = 1.0 - w_ceil

            mask_f = t_floor < B
            if mask_f.sum() > 0:
                np.add.at(voxel,
                          (t_floor[mask_f], y[mask_f], x[mask_f]),
                          pol[mask_f] * w_floor[mask_f])
            mask_c = t_ceil < B
            if mask_c.sum() > 0:
                np.add.at(voxel,
                          (t_ceil[mask_c], y[mask_c], x[mask_c]),
                          pol[mask_c] * w_ceil[mask_c])

        # 3채널 압축: [양극성, 음극성, 전체밀도]
        pos = np.clip(voxel, 0, None).sum(axis=0)
        neg = np.clip(-voxel, 0, None).sum(axis=0)
        mag = pos + neg

        def robust_norm(arr):
            nz = arr[arr > 0]
            if len(nz) == 0:
                return arr
            return np.clip(arr / (3.0 * nz.std() + 1e-8), 0, 1)

        rgb_like = np.stack(
            [robust_norm(pos), robust_norm(neg), robust_norm(mag)],
            axis=0
        )  # (3, ev_H, ev_W)

        # Resize: (352, 1216) -> (370, 1220)
        if rgb_like.shape[1:] != (self.img_H, self.img_W):
            rgb_like = np.stack([
                cv2.resize(rgb_like[c],
                           (self.img_W, self.img_H),
                           interpolation=cv2.INTER_LINEAR)
                for c in range(3)
            ], axis=0)

        # ImageNet 정규화
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        rgb_like = (rgb_like - mean) / std

        return torch.from_numpy(rgb_like.astype(np.float32))  # [3, H, W]

    def get_input_info(self, sequence, frame_id):
'''

    assert insert_before in src, "[ERROR] 패치 2 삽입 위치를 찾지 못함."
    src = src.replace(insert_before, new_method, 1)
    print("[완료] 패치 2: load_event_as_rgb 메서드 삽입")

    # ── 패치 3: get_input_info 본문 교체 ─────────────────────
    # 기존 get_input_info의 본문을 event 로딩으로 교체
    old_body = '''\
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

    new_body = '''\
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

    assert old_body in src, "[ERROR] 패치 3 교체 위치를 찾지 못함. 원본 파일을 확인하세요."
    src = src.replace(old_body, new_body, 1)
    print("[완료] 패치 3: get_input_info 교체")

    # 저장
    with open(TARGET, 'w') as f:
        f.write(src)

    print(f"\n✅ 패치 완료: {TARGET}")
    print(f"   원본 백업: {BACKUP}")
    print("\n다음 단계: 아래 명령으로 실험 0 실행")
    print("   cd /VoxFormer")
    print("   python tools/test.py \\")
    print("       projects/configs/voxformer/VoxFormer-T.py \\")
    print("       ckpts/voxformer-S.pth \\")
    print("       --eval ssc")


def verify_patch():
    """패치가 올바르게 적용됐는지 확인"""
    with open(TARGET) as f:
        src = f.read()
    checks = [
        ("event_root 설정",        "self.event_root"),
        ("load_event_as_rgb 메서드", "def load_event_as_rgb"),
        ("voxel 변환 코드",         "np.add.at(voxel"),
        ("get_input_info 교체",    "Event Voxel Grid를 로드"),
    ]
    print("\n패치 검증:")
    all_ok = True
    for name, keyword in checks:
        found = keyword in src
        status = "✅" if found else "❌"
        print(f"  {status} {name}")
        if not found:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("="*55)
    print("실험 0 패치 적용: Event Voxel Grid Naive Input")
    print("="*55)
    apply_patch()
    verify_patch()