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
        pts_channels=1024,  # the out of Point Transformer
        mid_channels=128,  # Multi level feature  first turn to
        out_channels=1024,
        img_levels=[0, 1, 2, 3],
        coord_type='DEPTH',
        align_corners=False,
        activate_out=True,
        fuse_out=False),
    backbone=dict(
        type='Point_Transformer',
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        out_channels=1024,
        channels=128,
        num_stages=4,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    decode_head=dict(
        type='Point_TransformerHead',
        in_channels=1024,
        mid_channels=512,
        channels=256,
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