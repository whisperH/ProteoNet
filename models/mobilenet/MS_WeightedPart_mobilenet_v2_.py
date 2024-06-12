# model settings
part_num=5
model_cfg = dict(
    model_name="WeightedPart_MBNet2",
    backbone=dict(type='MobilePartNetV2'),
    neck=dict(
        type='PartGlobalAveragePooling',
        part_num=part_num
    ),
    head=dict(
        type='WeightedPartClsHead',
        num_classes=4,
        in_channels=1280,
        part_num=part_num,
        loss=dict(type='focal_loss'),
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
        pretrained_weights = './datas/mobilenet_v2_.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = '/home/XXXX/code/MassSpectrumClsV1/logs/WeightedPart_MBNet2/2023-12-02-00-31-14fold4/Last_Epoch100.pth',
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

