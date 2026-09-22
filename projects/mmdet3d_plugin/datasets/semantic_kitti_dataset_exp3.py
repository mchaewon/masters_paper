"""
SemanticKittiDatasetExp3
=========================
Exp3: Dual Event Encoder (E_ego + E_obj) with Auxiliary Supervision

기존 SemanticKittiDatasetStage2 대비 변경사항:
  - get_input_info(): RGB only (변경 없음)
  - get_meta_info(): event_path, depth_path, moving_mask_path 추가
  - get_gt_info(): 변경 없음
"""

import os, glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from torchvision import transforms

# 기존 stage2 코드를 그대로 가져온 후 meta_info만 수정
# (실제 사용 시 stage2의 원본 파일에서 상속하거나 복사)
from .semantic_kitti_dataset_stage2 import SemanticKittiDatasetStage2


@DATASETS.register_module()
class SemanticKittiDatasetExp3(SemanticKittiDatasetStage2):
    """
    Exp3용 Dataset.
    img_metas에 event_path, depth_path, moving_mask_path를 추가한다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Exp3 추가 경로 설정
        data_root = self.data_root
        self.event_root       = os.path.join(data_root, "events", "sequences")
        self.depth_root       = os.path.join(data_root, "depth", "sequences")
        self.moving_mask_root = os.path.join(data_root, "moving_masks")

        # Event 파라미터
        self.num_event_bins = 5
        self.ev_H, self.ev_W = 352, 1216

        print(f"[Exp3 Dataset] event_root:       {self.event_root}")
        print(f"[Exp3 Dataset] depth_root:        {self.depth_root}")
        print(f"[Exp3 Dataset] moving_mask_root:  {self.moving_mask_root}")

    def get_meta_info(self, scan, sequence, frame_id, proposal_path):
        """기존 meta_info에 exp3용 경로 3개 추가"""
        meta_dict = super().get_meta_info(scan, sequence, frame_id, proposal_path)

        # Event voxel 경로
        # seq 00-03: events/sequences/{seq}/{frame}.npy
        # seq 04-10: events/{seq}/image_0/{frame}.npy (다른 패턴)
        ev_candidates = [
            os.path.join(self.event_root, sequence, f"{frame_id}.npy"),
            os.path.join(self.data_root, "events", sequence,
                         "image_0", f"{frame_id}.npy"),
        ]
        event_path = next((p for p in ev_candidates if os.path.exists(p)), None)

        # Depth 경로
        depth_path = os.path.join(
            self.depth_root, sequence, f"{frame_id}.npy"
        )
        if not os.path.exists(depth_path):
            depth_path = None

        # Moving mask 경로
        moving_mask_path = os.path.join(
            self.moving_mask_root, sequence, f"{frame_id}.npy"
        )
        if not os.path.exists(moving_mask_path):
            moving_mask_path = None

        meta_dict.update(dict(
            event_path       = event_path,
            depth_path       = depth_path,
            moving_mask_path = moving_mask_path,
            sequence_id      = sequence,
            frame_id         = frame_id,
        ))
        return meta_dict