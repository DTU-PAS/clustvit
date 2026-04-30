_base_ = [
    "../_base_/models/upernet_vit-b16_mask_k3.py",
    "../_base_/datasets/rumexweeds.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
default_scope = "mmseg"


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=3,
        channels=768,
    ),
)
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=8
)
val_dataloader = dict(batch_size=1)
