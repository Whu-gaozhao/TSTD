_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py',
    '../_base_/models/point_transformerV2.py',
    '../_base_/schedules/seg_cosine_50e.py', '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=12)
evaluation = dict(interval=2,save_best='miou')

# model settings
model = dict(
    backbone=dict(in_channels=6),  # [xyz, rgb]
    decode_head=dict(
        num_classes=13, ignore_index=13,
        loss_decode=dict(class_weight=None)),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=24))

# runtime settings
checkpoint_config = dict(interval=2)