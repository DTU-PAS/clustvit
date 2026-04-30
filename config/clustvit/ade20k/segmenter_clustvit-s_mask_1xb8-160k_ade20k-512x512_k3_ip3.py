_base_ = [
    "../../_base_/models/segmenter_clustvit-b16_mask_k3.py",
    "../../_base_/datasets/ade20k.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k.py",
]
default_scope = "mmseg"

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth"  # noqa

backbone_norm_cfg = dict(type="LN", eps=1e-6, requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
        cluster_mlp_size=1548,
        init_weights_zero=False,
        injection_points=[3],
    ),
    decode_head=dict(
        type="SegmenterMaskTransformerHead",
        in_channels=384,
        channels=384,
        num_classes=150,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
        ),
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
