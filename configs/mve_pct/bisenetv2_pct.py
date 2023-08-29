# model settings
model = dict(
    type='MultivewEncoderDecoder',
    img_backbone=dict(
        type='BiSeNetV2',
        detail_channels=(64, 64, 128),
        semantic_channels=(16, 32, 64, 128),
        semantic_expansion_ratio=6,
        bga_channels=128,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False),
    fusion_layer=dict(
        type='PointFusionMultiview',
        img_channels=[128, 16, 32, 64, 128],
        pts_channels=1024,  # the out of Point Transformer
        mid_channels=64,  # Multi level feature  first turn to
        out_channels=512,
        img_levels=[0, 1, 2, 3, 4],
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
        in_channels=512,
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