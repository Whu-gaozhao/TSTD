_base_ = [
    '../_base_/datasets/scannet_seg-3d-20class_block.py', '../_base_/models/pct.py',
    '../_base_/schedules/seg_cosine_100e.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=8)
evaluation = dict(interval=10, save_best='miou')
checkpoint_config = dict(interval=10)

# model settings
model = dict(
    decode_head=dict(
        num_classes=20,
        ignore_index=20,
        loss_decode=dict(class_weight=[
            2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
            4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
            5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
            5.3954206, 4.6971426
        ])),
    test_cfg=dict(
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))
