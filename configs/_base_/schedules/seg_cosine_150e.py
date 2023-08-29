# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
# optimizer = dict(type='SGD', lr=0.2, weight_decay=0.0001, momentum=0.9)
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=0.002)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=150)
