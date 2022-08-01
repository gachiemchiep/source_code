_base_ = [
    './datasets/tanachou.py',
    './models/efficientnet-b0-f4_lbs-mixup_tanachou.py'
]

# runtime
# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


evaluation = dict(interval=1, metric='accuracy', save_best='auto')
# work_dir = './work_dirs/try1/'

### optimizer
optimizer = dict(
    type='AdamW', lr=1e-4, weight_decay=0.3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1e-5,
    warmup_by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=10)