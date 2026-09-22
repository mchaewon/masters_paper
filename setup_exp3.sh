#!/bin/bash
# Exp3 셋업 스크립트
# 실행: bash setup_exp3.sh

set -e
VOXFORMER=/VoxFormer
DATASET=/VoxFormer/dataset/semantickitti

echo "========================================"
echo "Exp3 셋업 시작"
echo "========================================"

# # 1. 파일 복사
# echo "[1] 파일 복사..."
# cp voxformer_exp3.py \
#     $VOXFORMER/projects/mmdet3d_plugin/voxformer/detectors/voxformer_exp3.py

# cp semantic_kitti_dataset_exp3.py \
#     $VOXFORMER/projects/mmdet3d_plugin/datasets/semantic_kitti_dataset_exp3.py

# cp voxformer_exp3_config.py \
#     $VOXFORMER/projects/configs/voxformer/voxformer_exp3.py

# 2. __init__.py 등록
echo "[2] __init__.py 등록..."

python3 << 'PYEOF'
# detectors/__init__.py
path = "/VoxFormer/projects/mmdet3d_plugin/voxformer/detectors/__init__.py"
with open(path) as f:
    src = f.read()
line = "from .voxformer_exp3 import VoxFormerExp3\n"
if line not in src:
    src += line
    with open(path, 'w') as f:
        f.write(src)
    print("  ✅ detectors/__init__.py: VoxFormerExp3 등록")
else:
    print("  ℹ️  VoxFormerExp3 이미 등록됨")

# datasets/__init__.py
path = "/VoxFormer/projects/mmdet3d_plugin/datasets/__init__.py"
with open(path) as f:
    src = f.read()
line = "from .semantic_kitti_dataset_exp3 import SemanticKittiDatasetExp3\n"
if line not in src:
    src += line
    with open(path, 'w') as f:
        f.write(src)
    print("  ✅ datasets/__init__.py: SemanticKittiDatasetExp3 등록")
else:
    print("  ℹ️  SemanticKittiDatasetExp3 이미 등록됨")
PYEOF

# 3. Moving mask 전처리
echo "[3] Moving mask 전처리..."
cd $VOXFORMER
python precompute_moving_masks.py \
    --dataset_root $DATASET \
    --seqs 00 01 02 03 04 05 06 07 09 10 08 \
    --workers 4

# 4. 동작 확인
echo "[4] 동작 확인..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
import importlib
from mmcv import Config
cfg = Config.fromfile('projects/configs/voxformer/voxformer_exp3.py')
importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.voxformer.detectors import VoxFormerExp3
from projects.mmdet3d_plugin.datasets import SemanticKittiDatasetExp3
print("✅ 모든 클래스 임포트 성공")

from mmdet3d.models import build_detector
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
print("✅ 모델 빌드 성공")
n_ego = sum(p.numel() for p in model.ego_encoder.parameters())
n_obj = sum(p.numel() for p in model.obj_encoder.parameters())
print(f"  EgoEncoder params:  {n_ego:,}")
print(f"  ObjEncoder params:  {n_obj:,}")
PYEOF

echo ""
echo "========================================"
echo "셋업 완료!"
echo ""
echo "학습 실행:"
echo "  python tools/train.py \\"
echo "      projects/configs/voxformer/voxformer_exp3.py \\"
echo "      --work-dir results/exp3_dual_event"
echo ""
echo "Fine-tune (pretrained checkpoint 사용):"
echo "  python tools/train.py \\"
echo "      projects/configs/voxformer/voxformer_exp3.py \\"
echo "      --cfg-options load_from=ckpts/voxformer-T-3D/xxx.pth \\"
echo "      --work-dir results/exp3_finetune"
echo "========================================"