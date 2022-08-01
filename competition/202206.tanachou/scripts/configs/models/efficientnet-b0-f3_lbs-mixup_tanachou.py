### model definition
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet', arch='b0',
        frozen_stages=3,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth',
            prefix='backbone',
        )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=122,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            mode='classy_vision',
            num_classes=122),
        topk=(1, 5),
    ),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=122,
                      prob=1.)))