_base_ = [
    '../_base_/datasets/scannet_seg-3d-20class.py',
    '../_base_/models/mve_seg.py', '../_base_/schedules/seg_cosine_100e.py',
    '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=4)
evaluation = dict(interval=5, save_best='miou')
checkpoint_config = dict(interval=5)

# model settings
model = dict(
    backbone=dict(
        sa_cfg=dict(use_xyz=True), fu_cfg=dict(type='SelfAttentionFusion')),
    decode_head=dict(
        num_classes=20,
        ignore_index=20,
        num_heads=1,
        att_type="SelfAttention",
        align_corners=False,
        trans_with_mlp=False,
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