checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth"  # noqa
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
    type="DynSegEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type="ClustViT",
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
        num_clusters=3,
        injection_points=[4],
        cluster_mlp_size=3072,
        init_weights_zero=False,
    ),
    decode_head=dict(
        type="SegmenterMaskTransformerHead",
        in_channels=768,
        channels=768,
        num_classes=150,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="AuxPassthrough",
        loss_decode=[
            dict(
                type="ClustLoss",
                loss_name="clust_ce_loss",
                loss_weight=1.0,
                ignore_index=None,
            )
        ],
    ),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(480, 480)),
)
