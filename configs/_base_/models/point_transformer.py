# model settings
model = dict(
    type='EncoderDecoder3D',
    backbone=dict(
        type='PointTransformer',
        in_channels=6,
        embed_dim=32,
        num_point=8192,
        num_stage=4,
        num_neighbor=32,
        # neck_channel=512,
        transformer_channel=512,
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False),
    ),
    decode_head=dict(
        type='PointTransformerHead',
        embed_dim=32,
        num_stage=4,
        num_neighbor=16,
        transformer_channel=512,
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
