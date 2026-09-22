"""
전체 실험 환경 셋업 스크립트
=============================
실험별로 독립된 파일을 생성하여 코드 교체 없이 실험 전환 가능.

생성되는 파일:
  datasets/
    semantic_kitti_dataset_rgb.py        ← RGB only (baseline)
    semantic_kitti_dataset_exp1.py       ← RGB + Event concat [N,6,H,W]
    semantic_kitti_dataset_exp2.py       ← RGB only + event_path in metas

  detectors/
    voxformer_baseline.py                ← 기존 VoxFormer
    voxformer_exp1.py                    ← 6ch first conv
    voxformer_exp2.py                    ← EventEncoder + fusion

  configs/
    voxformer_baseline_scratch.py        ← RGB only scratch
    voxformer_exp1_scratch.py            ← Exp1 scratch
    voxformer_exp2_scratch.py            ← Exp2 scratch

실행:
    python setup_all_experiments.py
"""

import os, shutil

BASE_DATASET = "./projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_stage2.py"
BASE_DETECTOR = "./projects/mmdet3d_plugin/voxformer/detectors/voxformer.py"
BASE_CONFIG = "./projects/configs/voxformer/voxformer-T_deform3D.py"

DATASET_DIR = "./projects/mmdet3d_plugin/datasets"
DETECTOR_DIR = "./projects/mmdet3d_plugin/voxformer/detectors"
CONFIG_DIR = "./projects/configs/voxformer"


# ============================================================
# 원본(RGB only) dataset 코드
# ============================================================
RGB_GET_INPUT = '''    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img [N, 3, img_H, img_W]
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        rgb_path = os.path.join(
            self.data_root,  "sequences", sequence, "image_2", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]
        image_list.append(self.normalize_rgb(img))

        for i in self.target_frames:
            id = int(frame_id)
            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)
            rgb_path = os.path.join(
                self.data_root,  "sequences", sequence, "image_2", target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]
            image_list.append(self.normalize_rgb(img))

        image_tensor = torch.stack(image_list, dim=0)  # [N, 3, 370, 1220]
        return image_tensor
'''

# Exp1: 6채널 concat
EXP1_INIT_EXTRA = '''
        # Exp1: Event 설정
        self.event_root = os.path.join(data_root, "events", "sequences")
        self.num_event_bins = 5
        self.ev_H, self.ev_W = 352, 1216
'''

EXP1_LOAD_EVENT = '''
    def load_event_as_rgb(self, sequence, frame_id):
        """Raw events -> 3채널 [3, img_H, img_W]"""
        import cv2
        event_path = os.path.join(self.event_root, sequence, f"{frame_id}.npy")
        if not os.path.exists(event_path):
            return torch.zeros(3, self.img_H, self.img_W)
        events = np.load(event_path)
        B, H, W = self.num_event_bins, self.ev_H, self.ev_W
        voxel = np.zeros((B, H, W), dtype=np.float32)
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2].astype(np.float64)
        p = events[:, 3].astype(np.float32)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y, t, p = x[ok], y[ok], t[ok], p[ok]
        if len(t) > 0:
            t_min, t_max = t.min(), t.max()
            tn = (t - t_min) / (t_max - t_min + 1e-8) * (B - 1)
            pol = np.where(p > 0, 1.0, -1.0).astype(np.float32)
            tf = tn.astype(np.int32)
            tc = tf + 1
            wc = (tn - tf).astype(np.float32)
            wf = 1.0 - wc
            mf = tf < B
            if mf.sum(): np.add.at(voxel, (tf[mf], y[mf], x[mf]), pol[mf]*wf[mf])
            mc = tc < B
            if mc.sum(): np.add.at(voxel, (tc[mc], y[mc], x[mc]), pol[mc]*wc[mc])
        pos = np.clip(voxel, 0, None).sum(0)
        neg = np.clip(-voxel, 0, None).sum(0)
        mag = pos + neg
        def norm(a):
            nz = a[a > 0]
            return np.clip(a / (3.0 * nz.std() + 1e-8), 0, 1) if len(nz) else a
        rgb_like = np.stack([norm(pos), norm(neg), norm(mag)], 0)
        if rgb_like.shape[1:] != (self.img_H, self.img_W):
            rgb_like = np.stack([
                cv2.resize(rgb_like[c], (self.img_W, self.img_H), cv2.INTER_LINEAR)
                for c in range(3)
            ], 0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        rgb_like = (rgb_like - mean) / std
        return torch.from_numpy(rgb_like.astype(np.float32))

'''

EXP1_GET_INPUT = '''    def get_input_info(self, sequence, frame_id):
        """[Exp1] RGB(3ch) + Event(3ch) concat -> [N, 6, H, W]"""
        seq_len = len(self.poses[sequence])
        image_list = []

        def load_rgb(seq, fid):
            p = os.path.join(self.data_root, "sequences", seq, "image_2", fid+".png")
            img = Image.open(p).convert("RGB")
            if self.color_jitter: img = self.color_jitter(img)
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            return img[:self.img_H, :self.img_W, :]

        rgb = load_rgb(sequence, frame_id)
        ev  = self.load_event_as_rgb(sequence, frame_id)
        image_list.append(torch.cat([self.normalize_rgb(rgb), ev], dim=0))  # [6,H,W]

        for i in self.target_frames:
            id = int(frame_id)
            tid = frame_id if (id+i < 0 or id+i > seq_len-1) else str(id+i).zfill(6)
            rgb = load_rgb(sequence, tid)
            ev  = self.load_event_as_rgb(sequence, tid)
            image_list.append(torch.cat([self.normalize_rgb(rgb), ev], dim=0))

        return torch.stack(image_list, dim=0)  # [5, 6, 370, 1220]
'''

# Exp2: img_metas에 event_path 추가
EXP2_META_ADDITION = '''        event_path = os.path.join(self.event_root, sequence, f"{frame_id}.npy")
        '''

EXP2_META_DICT_ADDITION = '''            event_path = event_path,
        '''


def make_dataset_file(tag, class_name, get_input_code,
                      init_extra="", extra_methods="",
                      meta_extra_before="", meta_dict_extra=""):
    """기존 dataset 파일을 읽어 수정된 버전 생성"""
    # 원본 백업에서 읽기
    orig = BASE_DATASET.replace(".py", "_orig.py")
    if not os.path.exists(orig):
        print(f"⚠️  원본 백업 없음: {orig}")
        orig = BASE_DATASET

    with open(orig) as f:
        src = f.read()

    # 클래스 이름 변경
    src = src.replace(
        "class SemanticKittiDatasetStage2(Dataset):",
        f"class {class_name}(Dataset):"
    )
    src = src.replace(
        "@DATASETS.register_module()\nclass SemanticKittiDatasetStage2",
        f"@DATASETS.register_module()\nclass {class_name}"
    )

    # __init__ 추가 설정
    if init_extra:
        src = src.replace(
            "        self.img_W = 1220\n        self.img_H = 370",
            f"        self.img_W = 1220\n        self.img_H = 370\n{init_extra}"
        )

    # get_input_info 교체
    start = src.find("    def get_input_info(self, sequence, frame_id):")
    end   = src.find("\n    def ", start + 1)
    if start >= 0 and end >= 0:
        src = src[:start] + get_input_code + "\n" + src[end+1:]

    # 메서드 추가 (get_input_info 바로 앞)
    if extra_methods:
        ins = src.find("    def get_input_info")
        src = src[:ins] + extra_methods + src[ins:]

    # meta_dict에 event_path 추가
    if meta_dict_extra:
        old_meta = "            img_shape = [(self.img_H,self.img_W)]\n        )"
        new_meta = "            img_shape = [(self.img_H,self.img_W)],\n" + \
                   meta_dict_extra + "\n        )"
        src = src.replace(old_meta, new_meta, 1)

    if meta_extra_before:
        old_trigger = "        meta_dict = dict("
        src = src.replace(old_trigger, meta_extra_before + old_trigger, 1)

    out = os.path.join(DATASET_DIR, f"semantic_kitti_dataset_{tag}.py")
    with open(out, 'w') as f:
        f.write(src)
    print(f"✅ {out}")
    return out


def make_detector_exp1():
    """Exp1: 6채널 first conv"""
    orig = BASE_DETECTOR.replace(".py", "_orig.py")
    if not os.path.exists(orig):
        orig = BASE_DETECTOR
    with open(orig) as f:
        src = f.read()

    # import nn 추가
    src = src.replace(
        "import time\nimport copy\nimport torch\nimport numpy as np",
        "import time\nimport copy\nimport torch\nimport torch.nn as nn\nimport numpy as np"
    )

    # __init__ super() 뒤에 6ch 확장 추가
    old_super = """        super(VoxFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)"""
    new_super = old_super + """

        # Exp1: 첫 conv 3ch -> 6ch (RGB pretrained + Event zero-init)
        self._expand_first_conv_to_6ch()"""
    src = src.replace(old_super, new_super, 1)

    # 메서드 추가
    expand_method = """
    def _expand_first_conv_to_6ch(self):
        first_conv = self.img_backbone.conv1
        if first_conv.in_channels == 6:
            return
        old_w = first_conv.weight.data.clone()
        new_conv = nn.Conv2d(6, first_conv.out_channels,
                             first_conv.kernel_size, first_conv.stride,
                             first_conv.padding, bias=False)
        new_conv.weight.data[:, :3] = old_w
        new_conv.weight.data[:, 3:] = 0.0
        self.img_backbone.conv1 = new_conv
        print("[Exp1] First conv: 3ch -> 6ch")

"""
    src = src.replace("    def extract_img_feat",
                      expand_method + "    def extract_img_feat", 1)

    out = os.path.join(DETECTOR_DIR, "voxformer_exp1.py")
    with open(out, 'w') as f:
        f.write(src)
    print(f"✅ {out}")
    return out


def make_detector_exp2():
    """Exp2: EventEncoder + feature fusion"""
    orig = BASE_DETECTOR.replace(".py", "_orig.py")
    if not os.path.exists(orig):
        orig = BASE_DETECTOR
    with open(orig) as f:
        src = f.read()

    src = src.replace(
        "import time\nimport copy\nimport torch\nimport numpy as np",
        "import time\nimport copy\nimport torch\nimport torch.nn as nn\nimport numpy as np"
    )

    old_super = """        super(VoxFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)"""
    new_super = old_super + """

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
        print("[Exp2] EventEncoder + FusionConv initialized")"""
    src = src.replace(old_super, new_super, 1)

    event_methods = """
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

"""
    src = src.replace("    def extract_img_feat",
                      event_methods + "    def extract_img_feat", 1)

    # forward_train, forward_test에 fusion 삽입
    for old_extract, new_extract in [
        ("        img_feats = self.extract_feat(img=img) \n        losses = dict()",
         "        img_feats = self.extract_feat(img=img)\n        img_feats = self._fuse_event_feats(img_feats, img_metas, img_H, img_W)\n        losses = dict()"),
        ("        img_feats = self.extract_feat(img=img) \n        outs = self.pts_bbox_head",
         "        img_feats = self.extract_feat(img=img)\n        img_feats = self._fuse_event_feats(img_feats, img_metas, img_H, img_W)\n        outs = self.pts_bbox_head"),
    ]:
        if old_extract in src:
            src = src.replace(old_extract, new_extract, 1)

    # forward_test return fix
    old_ret = "        return completion_results"
    new_ret = """        if isinstance(completion_results, dict):
            completion_results = [completion_results]
        return completion_results"""
    src = src.replace(old_ret, new_ret, 1)

    out = os.path.join(DETECTOR_DIR, "voxformer_exp2.py")
    with open(out, 'w') as f:
        f.write(src)
    print(f"✅ {out}")
    return out


def make_config(tag, dataset_class, detector_type, work_dir):
    """실험별 config 파일 생성"""
    with open(BASE_CONFIG) as f:
        src = f.read()

    # pretrained 제거 (scratch)
    src = src.replace(
        "   pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),",
        "   pretrained=None,"
    )

    # work_dir 변경
    old_wd = src.split('\n')[0]  # 첫 줄
    src = src.replace(old_wd, f"work_dir = '{work_dir}'", 1)

    # dataset type 변경
    src = src.replace(
        "dataset_type = 'SemanticKittiDatasetStage2'",
        f"dataset_type = '{dataset_class}'"
    )

    # detector type 변경 (Exp1, Exp2만)
    if detector_type != 'VoxFormer':
        src = src.replace(
            "   type='VoxFormer',",
            f"   type='{detector_type}',"
        )

    # Exp1: FPN in_channels 유지 (backbone output은 동일)
    # (first conv만 바뀌고 최종 출력은 1024로 동일)

    out = os.path.join(CONFIG_DIR, f"voxformer_{tag}_scratch.py")
    with open(out, 'w') as f:
        f.write(src)
    print(f"✅ {out}")
    return out


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("전체 실험 환경 셋업")
    print("=" * 60)

    # 원본 백업 확인
    orig = BASE_DATASET.replace(".py", "_orig.py")
    if not os.path.exists(orig):
        shutil.copy2(BASE_DATASET, orig)
        print(f"[백업] {orig}")

    orig_det = BASE_DETECTOR.replace(".py", "_orig.py")
    if not os.path.exists(orig_det):
        shutil.copy2(BASE_DETECTOR, orig_det)
        print(f"[백업] {orig_det}")

    # 1. Dataset 파일 생성
    print("\n--- Dataset 파일 생성 ---")
    make_dataset_file("rgb",  "SemanticKittiDatasetRGB",
                      RGB_GET_INPUT)
    make_dataset_file("exp1", "SemanticKittiDatasetExp1",
                      EXP1_GET_INPUT,
                      init_extra=EXP1_INIT_EXTRA,
                      extra_methods=EXP1_LOAD_EVENT)
    make_dataset_file("exp2", "SemanticKittiDatasetExp2",
                      RGB_GET_INPUT,
                      init_extra=EXP1_INIT_EXTRA,
                      meta_extra_before=f"        event_path = os.path.join(self.event_root, sequence, f'{{frame_id}}.npy')\n",
                      meta_dict_extra="            event_path = event_path,")

    # 2. Detector 파일 생성
    print("\n--- Detector 파일 생성 ---")
    make_detector_exp1()
    make_detector_exp2()

    # 3. Config 파일 생성
    print("\n--- Config 파일 생성 ---")
    make_config("baseline", "SemanticKittiDatasetRGB",
                "VoxFormer",    "results/baseline_scratch")
    make_config("exp1",     "SemanticKittiDatasetExp1",
                "VoxFormerExp1","results/exp1_scratch")
    make_config("exp2",     "SemanticKittiDatasetExp2",
                "VoxFormerExp2","results/exp2_scratch")

    # 4. __init__.py 업데이트 안내
    print("""
--- 추가로 해야 할 것 ---
1. detectors/__init__.py에 새 클래스 등록:
   grep -n "VoxFormer" ./projects/mmdet3d_plugin/voxformer/detectors/__init__.py

2. datasets/__init__.py에 새 클래스 등록:
   grep -n "SemanticKitti" ./projects/mmdet3d_plugin/datasets/__init__.py
""")

    print("""
=== 실행 커맨드 ===

# Baseline (RGB only scratch)
python tools/train.py \\
    projects/configs/voxformer/voxformer_baseline_scratch.py \\
    --work-dir results/baseline_scratch

# Exp 1 (RGB+Event naive concat scratch)
python tools/train.py \\
    projects/configs/voxformer/voxformer_exp1_scratch.py \\
    --work-dir results/exp1_scratch

# Exp 2 (RGB+Event dedicated encoder scratch)
python tools/train.py \\
    projects/configs/voxformer/voxformer_exp2_scratch.py \\
    --work-dir results/exp2_scratch
""")