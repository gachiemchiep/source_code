#### runtime
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'



#### model
dict_file = '/home/user/workspace/2022.toppan_ocr/configs/rec/toppan_dict.txt'
label_convertor = dict(
    type='CTCConvertor', dict_file=dict_file, with_unknown=False)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)


#### pipeline

img_norm_cfg = dict(mean=[127], std=[127])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=800,
        max_width=800,
        keep_aspect_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'resize_shape', 'valid_ratio', 'img_norm_cfg',
            'ori_filename', 'img_shape', 'ori_shape'
        ]),
]

# load dataset
prefix = '/home/user/workspace/2022.toppan_ocr/dataset/toppan_ocr_2'
train = dict(
    type='OCRDataset',
    img_prefix=f'{prefix}',
    ann_file=f'{prefix}/instances_train.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test = dict(
    type='OCRDataset',
    img_prefix=f'{prefix}',
    ann_file=f'{prefix}/instances_test.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

train_list = [train]
# train_list = [test]
test_list = [test]

### lr policy

# optimizer
# learning policy
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[16, 18])
total_epochs = 20
# running settings
checkpoint_config = dict(interval=1)



data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True