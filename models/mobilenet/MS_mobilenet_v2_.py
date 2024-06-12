# model settings

model_cfg = dict(
    model_name="MBNet2",
    backbone=dict(type='MobilePartNetV2'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

# train
data_cfg = dict(
    batch_size = 32,
    num_workers = 4,
    train = dict(
        pretrained_flag = False,
        pretrained_weights = './pretrain/mobilenet_v2-b0353104.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = '/home/XXXX/code/MassSpectrumClsV1/logs/MBNet2/2023-12-02-00-31-03fold4/Last_Epoch100.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)
# batch 32
# lr = 0.045 *32 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.045 * 32/256,
    momentum=0.9,
    weight_decay=0.00004)

# learning 
lr_config = dict(type='StepLrUpdater', step=1, gamma=0.98)

