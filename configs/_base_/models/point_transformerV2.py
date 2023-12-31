model = dict(
    type='EncoderDecoder3D',
    backbone=dict(
        type='PointTransFormerV2',
        in_channels=6,
        # in_channels=6,
        #   num_classes,
        patch_embed_depth=2,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=(2,2,6),
        enc_channels=(96,192,384),
        enc_groups=(12,24,48),
        enc_neighbours=(16,16,16),
        grid_sizes=(0.1, 0.2, 0.4),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        enable_checkpoint=False,
    ),
    decode_head=dict(
        type='PointTransformerV2Head',
        channels=48,
        num_classes=13,
        enc_channels=(96,192,384),
        dec_channels=(48,96,192),
        dec_depths=(1,1,1),
        dec_groups=(6,12,24),
        dec_neighbours=(16,16,16),
        attn_qkv_bias = True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="interp",
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide')
)