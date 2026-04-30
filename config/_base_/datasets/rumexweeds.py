# dataset settings

dataset_type = 'RumexWeedsDataset'
data_root = 'data/RumexWeeds/processed'
crop_size = (512, 512)
CLASSES = ['background', 'rumex_obtusifolius', 'rumex_crispus']
PALETTE = [
    [0, 0, 0],        # background
    [0, 255, 0],      # rumex_obtusifolius
    [0, 0, 255],      # rumex_crispus
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        mask_dir='train/masks',
        split_file=None,  # use all images in train/images
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        mask_dir='val/masks',
        split_file=None,  # use all images in val/images
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        mask_dir='test/masks',
        split_file=None,  # use all images in test/images
        pipeline=test_pipeline,
    ),
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator 