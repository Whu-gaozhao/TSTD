# optimizer
# This schedule is mainly used on ScanNet dataset in segmentation task
# adpot configs in mvp net except lr
# optimizer = dict(type='SGD', lr=0.2, weight_decay=0.0001, momentum=0.9)
optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
