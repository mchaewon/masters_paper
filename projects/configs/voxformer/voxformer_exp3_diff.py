"""
Exp3 Config: Dual Event Encoder with Auxiliary Supervision
==========================================================
기존 voxformer-T_deform3D.py 기반, Exp3용 변경사항만 적용
"""

_base_ = ['./voxformer-T_deform3D.py']

work_dir = 'results/exp3_dual_event'

# ── Model: VoxFormerExp3 사용 ────────────────────────────────
model = dict(
    # type='VoxFormerExp3',
    type='VoxFormerExp3Diff',  # GN 사용 시
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    # 기존 backbone, neck, head는 그대로 (_base_에서 상속)

    # Exp3 추가 파라미터
    event_channels=128,    # FPN output과 동일
    num_event_bins=5,
    lambda_ego=0.1,        # depth aux loss 가중치
    lambda_obj=0.1,        # dynamic mask aux loss 가중치
    img_H=370,
    img_W=1220,
    ev_H=352,
    ev_W=1216,
)

# ── Dataset: SemanticKittiDatasetExp3 사용 ───────────────────
dataset_type = 'SemanticKittiDatasetExp3'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(type=dataset_type),
    val=dict(type=dataset_type),
    test=dict(type=dataset_type),
)

# ── Checkpoint 저장 ──────────────────────────────────────────
checkpoint_config = dict(_delete_=True, interval=1, max_keep_ckpts=3)

# ── 학습 설정 ────────────────────────────────────────────────
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            # Ego/Obj encoder: 더 높은 lr (처음부터 학습)
            'ego_encoder': dict(lr_mult=2.0),
            'obj_encoder': dict(lr_mult=2.0),
            'event_fusion': dict(lr_mult=2.0),
            # RGB backbone: 낮은 lr (pretrained 유지)
            'img_backbone': dict(lr_mult=0.1),
        }
    )
)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')]
)