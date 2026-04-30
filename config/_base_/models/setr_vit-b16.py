checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth"  # pretrained ViT base

# model settings
backbone_norm_cfg = dict(type="LN", eps=1e-6, requires_grad=True)

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type="VisionTransformer",
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode="bicubic",
        out_indices=(3, 6, 9, 11),  # layers for multi-level features
    ),
    decode_head=dict(
        type="SETRMLAHead",
        in_channels=[768, 768, 768, 768],
        in_index=(0, 1, 2, 3),
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(480, 480)),
)
