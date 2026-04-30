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
    backbone=dict(
        type="CTSVisionTransformer",
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        with_cp=False,
        policy_method='policy_net',
        policy_schedule=(612, 103),  # 30% token reduction
        policynet_ckpt='weights/policynet.pth',  # Update this path
        out_indices=(11,),
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
    auxiliary_head=None,  # CTS doesn't need auxiliary head like ClustViT
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(480, 480)),
) 