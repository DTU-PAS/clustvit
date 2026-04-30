_base_ = [
    "../_base_/models/upernet_vit-b16_mask_k3.py",
    "../_base_/datasets/suim.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
default_scope = "mmseg"
backbone_norm_cfg = dict(type="LN", eps=1e-6, requires_grad=True)

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="CTSVisionTransformer",
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode="bicubic",
        policy_method='policy_net',
        policy_schedule=(612, 103),
        policynet_ckpt='weights/policynet.pth',
        out_indices=[3, 5, 7, 11],
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=9,
        channels=768,
    ),
    auxiliary_head=None,
)
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer)
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1) 


