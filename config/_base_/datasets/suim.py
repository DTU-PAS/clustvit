# dataset settings

CLASSES = [
    'background', 'water', 'sky', 'sand', 'coral', 'plant', 'fish', 'wreck', 'human'
]
PALETTE = [
    [0, 0, 0],        # background
    [0, 0, 255],      # water
    [0, 255, 255],    # sky
    [194, 178, 128],  # sand
    [255, 0, 0],      # coral
    [0, 255, 0],      # plant
    [255, 255, 0],    # fish
    [255, 0, 255],    # wreck
    [128, 0, 128],    # human
]

dataset_type = 'SuimDataset'
data_root = 'data/suim'
crop_size = (512, 512)
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
        img_dir='train_val/images',
        mask_dir='train_val/masks_label',
        split_file='train_val/train.txt',
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
        img_dir='train_val/images',
        mask_dir='train_val/masks_label',
        split_file='train_val/val.txt',
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
        mask_dir='test/masks_label',
        split_file=None,  # use all images in test/images
        pipeline=test_pipeline,
    ),
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator 