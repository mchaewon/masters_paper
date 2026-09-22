_base_ = ['./voxformer_exp3.py']

work_dir = 'results/exp3_bc'

_dim_             = 128
_pos_dim_         = 64   # _dim_//2
_ffn_dim_         = 256  # _dim_*2
_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_layers_self_  = 2
_num_points_self_  = 8
_num_levels_       = 1
_num_cams_         = 5
point_cloud_range  = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size         = [0.2, 0.2, 0.2]

model = dict(
    type='VoxFormerExp3BC',
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    event_channels=128,
    num_event_bins=5,
    num_groups=8,
    lambda_ego=0.1,
    lambda_obj=0.1,
    img_H=370, img_W=1220,
    ev_H=352, ev_W=1216,

    pts_bbox_head=dict(
        _delete_=True,
        type='VoxFormerHeadBC',
        bev_h=128, bev_w=128, bev_z=16,
        embed_dims=_dim_,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        cross_transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams=_num_cams_,
            encoder=dict(
                type='VoxFormerEncoder',
                num_layers=_num_layers_cross_,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[dict(
                        type='DeformCrossAttention',
                        pc_range=point_cloud_range,
                        num_cams=_num_cams_,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=_num_points_cross_,
                            num_levels=_num_levels_),
                        embed_dims=_dim_)],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn','norm','ffn','norm')))),
        self_transformer=dict(
            type='PerceptionTransformer3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams=_num_cams_,
            encoder=dict(
                type='VoxFormerEncoder3D',
                num_layers=_num_layers_self_,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer3D',
                    attn_cfgs=[dict(
                        type='DeformSelfAttention3DCustom',
                        embed_dims=_dim_,
                        num_levels=1,
                        num_points=_num_points_self_)],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn','norm','ffn','norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=512,
            col_num_embed=512),
        # VoxFormerHeadBC 추가 파라미터
        use_event_query=False,
        use_dual_stream=True,
        event_top_k=300,
        event_attn_layers=2,
        event_attn_heads=4,
    ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4)),
)

checkpoint_config = dict(_delete_=True, interval=1, max_keep_ckpts=3)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone':       dict(lr_mult=0.0),
            'img_neck':           dict(lr_mult=0.0),
            'cross_transformer':  dict(lr_mult=0.0),
            'self_transformer':   dict(lr_mult=0.0),
            'bev_embed':          dict(lr_mult=0.0),
            'mask_embed':         dict(lr_mult=0.0),
            'positional_encoding':dict(lr_mult=0.0),
            'header':             dict(lr_mult=0.0),
            'ego_encoder':        dict(lr_mult=1.0),
            'obj_encoder':        dict(lr_mult=1.0),
            'event_2d_proj':      dict(lr_mult=1.0),
            'event_cross_attn':   dict(lr_mult=1.0),
            'gated_fusion_3d':    dict(lr_mult=1.0),
        }
    ))
