#### dataset settings

dataset_type = 'CustomDataset'

_base_ = ['./randAug.py']
# remove background
# then paste on random image with random rotated and translation
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
            type="RepeatDataset", times=1, 
            dataset=dict(
                type=dataset_type,
                data_prefix='../data/train_aug',
                pipeline={{_base_.train_pipeline}} )),
    val=dict(
        type=dataset_type,
        data_prefix='../data/train',
        pipeline={{_base_.test_pipeline}},
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='../data/train',
        pipeline={{_base_.test_pipeline}},
        test_mode=True))