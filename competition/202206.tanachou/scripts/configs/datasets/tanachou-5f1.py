#### dataset settings
# 2-fold : fold count = 0
dataset_type = 'CustomDataset'

_base_ = ['./randAug.py']

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type="RepeatDataset", times=50, 
        dataset=dict(
            type='KFoldDataset',
            num_splits=5, 
            fold=1, 
            dataset=dict(
                type=dataset_type,
                data_prefix='../data/train',
                pipeline={{_base_.train_pipeline}} ))),
    val=dict(
        type="KFoldDataset",
        dataset=dict(
            type=dataset_type,
            data_prefix='../data/train',
            pipeline={{_base_.test_pipeline}},
        ),
        num_splits=5, 
        fold=1, 
        test_mode=True),
    test=dict(
        type="KFoldDataset",
        dataset=dict(
            type=dataset_type,
            data_prefix='../data/train',
            pipeline={{_base_.test_pipeline}},
        ),
        num_splits=2, 
        fold=0, 
        test_mode=True),
    )