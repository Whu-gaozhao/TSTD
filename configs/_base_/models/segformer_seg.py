# model settings
model = dict(
    type='MultivewEncoderDecoder',
    img_backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    fusion_layer=dict(
        type='PointFusionMultiview',
        img_channels=[32, 64, 160, 256],
        # pts_channels=1024,  # the out of Point Transformer
        pts_channels=1024,  # the out of Point Transformer
        mid_channels=128,  # Multi level feature  first turn to
        out_channels=1024,
        img_levels=[0, 1, 2, 3],
        coord_type='DEPTH',
        align_corners=False,
        activate_out=True,
        fuse_out=False),
    backbone=dict(
        type='MVE_PCT',
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        num_points=(1024, 256, 64, 16),
        radius=(0.1, 0.2, 0.4, 0.8),
        # num_samples=(32, 32, 32, 32),
        num_samples=(64, 64, 64, 64),
        # sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256,
        #                                                             512)),
        sa_channels=((64, 64, 128), (128, 128, 256), (256, 256, 512), (512, 512,
                                                                    1024)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='TwoStreamSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False),
        fu_cfg=dict(type='SelfAttentionFusion'),
    ),
    neck=dict(
        type='MveSegNECK',
        # fp_channels=((768, 256, 256), (384, 256, 256),
        #                       (320, 256, 128), (128, 128, 128, 128)),
        fp_channels=((1536, 2048, 2048), (2304, 2048, 2048),(2176,2048,1024)
                             ),
        # fp_channels=((768, 512, 1024),),
        fp_norm_cfg=dict(type='BN2d'),
        ignore_first_skip=True,
        late_fuse=False,

    ),
    decode_head=dict(
        type='MVE_SegHead',
        fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
                     (128, 128, 128, 128)),
        channels=128,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))