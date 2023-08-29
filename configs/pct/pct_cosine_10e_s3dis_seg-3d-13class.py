_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py', '../_base_/models/pct.py',
    '../_base_/schedules/seg_cosine_10e.py', '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=16)
evaluation = dict(interval=1, save_best='miou')

# optimizer settings
optimizer = dict(type='SGD', lr=0.05, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

# model settings
model = dict(
    backbone=dict(
        in_channels=9,
        out_channels=1024,
    ),  # [xyz, rgb, normalized_xyz]
    decode_head=dict(
        num_classes=13, ignore_index=13,
        loss_decode=dict(class_weight=None)),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=24,
        mode='slide'))

# runtime settings
checkpoint_config = dict(interval=1)
