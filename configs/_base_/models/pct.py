# model settings
model = dict(
    type='EncoderDecoder3D',
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