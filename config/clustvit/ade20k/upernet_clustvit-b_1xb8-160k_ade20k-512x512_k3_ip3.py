_base_ = [
    "../../_base_/models/upernet_vit-b16_mask_k3.py",
    "../../_base_/datasets/ade20k.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k.py",
]
default_scope = "mmseg"


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    type="DynSegEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="ClustViT",
        num_clusters=3,
        injection_points=[3],
        cluster_mlp_size=3072,
        init_weights_zero=False,
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=150,
        channels=768,
    ),
    auxiliary_head=dict(
        type="AuxPassthrough",
        loss_decode=[
            dict(
                type="ClustLoss",
                loss_name="clust_ce_loss",
                loss_weight=0.1,
                ignore_index=None,
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
