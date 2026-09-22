# Exp3 Diff config 기반, detector/head type만 변경
_base_ = ['./voxformer_exp3_diff.py']

model = dict(
    type='VoxFormerExp3Hybrid',   # VoxFormerExp3Diff → VoxFormerExp3Hybrid

    pts_bbox_head=dict(
        _delete_=True,
        type='VoxFormerHeadEvent',  # VoxFormerHead → VoxFormerHeadEvent
        use_event_3d=True,

        # 아래는 voxformer_exp3_diff.py의 pts_bbox_head와 완전히 동일
        bev_h=128, bev_w=128, bev_z=16,
        embed_dims=128,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        cross_transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=5,
            encoder=dict(
                type='VoxFormerEncoder',
                num_layers=3,
                pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[dict(
                        type='DeformCrossAttention',
                        pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                        num_cams=5,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=128,
                            num_points=8,
                            num_levels=1),
                        embed_dims=128)],
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn','norm','ffn','norm')))),
        self_transformer=dict(
            type='PerceptionTransformer3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=5,
            encoder=dict(
                type='VoxFormerEncoder3D',
                num_layers=2,
                pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer3D',
                    attn_cfgs=[dict(
                        type='DeformSelfAttention3DCustom',
                        embed_dims=128,
                        num_levels=1,
                        num_points=8)],
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn','norm','ffn','norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=64,
            row_num_embed=512,
            col_num_embed=512),
    ),
)

evaluation = dict(interval=1, save_best='ssc_SemanticKITTI/mIoU')
