_base_ = [
    "../../_base_/models/segmenter_clustvit-b16_mask_k3.py",
    "../../_base_/datasets/ade20k.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k.py",
]
default_scope = "mmseg"


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        cluster_mlp_size=3096,
        init_weights_zero=False,
        injection_points=[3],
    ),
    auxiliary_head=dict(
        type="AuxPassthrough",
        loss_decode=[
            dict(
                type="ClustLoss",
                loss_name="clust_ce_loss",
                loss_weight=0.1,
                ignore_index=-100,
            )
        ],
    ),
)
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=8
)
val_dataloader = dict(batch_size=1)
