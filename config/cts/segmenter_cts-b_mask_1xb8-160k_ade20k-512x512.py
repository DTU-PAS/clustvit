_base_ = [
    "../_base_/models/cts_vit.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]
default_scope = "mmseg"

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        policy_schedule=(612, 103),  # 30% token reduction
        policynet_ckpt='weights/policynet.pth',  # Update this path
    ),
    auxiliary_head=None,
)
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer)
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1) 