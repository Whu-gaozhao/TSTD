_base_ = [
    '../_base_/datasets/scannet_seg-3d_23dim.py',
    '../_base_/schedules/seg_cosine_200e.py', '../_base_/default_runtime.py'
]

dataset_type = 'ScanNetSegDataset'
data_root = 'G:/Datasets/scannet/'
class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=23,
        use_dim=23),
    dict(
        # a wrapper in order to successfully call test function
        # actually we don't perform test-time-aug
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]               
# data settings
data = dict(
    samples_per_gpu=16,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'segformer_avg/scannet_infos_val.pkl',
        # ann_file=data_root + '056800/056800.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names)))
evaluation = dict(interval=5)

# model settings
model = dict(
    type='BlankModel',
    num_classes=20,
    test_cfg=dict(
        mode='slide',
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))

# runtime settings
checkpoint_config = dict(interval=5)
