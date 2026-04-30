"""
Content-aware Token Sharing (CTS) Vision Transformer Backbone for MMSegmentation
Adapted from https://github.com/tue-mps/cts-segmenter
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import _load_weights
import os
from urllib.parse import urlparse


# Content-aware Token Sharing (CTS) Vision Transformer Backbone for MMSegmentation
# Extends MMSegmentation's VisionTransformer with PolicyNet and CTS logic
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.backbones.vit import VisionTransformer
from mmengine.registry import MODELS

# --- PolicyNet and CTS utilities (as before) ---
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_lite0', pretrained=True, features_only=True)
        self.head = nn.Conv2d(320, 2, 1)
    def forward(self, x):
        feats = self.backbone(x)
        x = self.head(feats[-1])
        return {'logits': x}

def policy_indices_by_policynet_pred(images, patch_size, policy_schedule, policynet_pred):
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    clue = policynet_pred['logits'].to(torch.float)
    base_grid_H, base_grid_W = H // (patch_size * 2), W // (patch_size * 2)
    num_scale_1, num_scale_2 = policy_schedule
    group_scores = torch.softmax(clue, dim=1)[:, 1]
    selected_msk_scale_1_per_img = []
    selected_msk_scale_2_per_img = []
    group_scores = rearrange(group_scores, 'b h w-> b (h w)')
    group_scores_sorted, group_scores_idx = torch.sort(group_scores, descending=True, dim=1)
    for b in range(B):
        # --- Scale 2 mask selection ---
        grouped_mask = torch.zeros((base_grid_H, base_grid_W), dtype=torch.bool, device=images.device)
        grouped_mask = rearrange(grouped_mask, 'h w-> (h w)')
        group_scores_idx_selected = group_scores_idx[b, :num_scale_2]
        grouped_mask[group_scores_idx_selected] = True
        grouped_mask = grouped_mask.view((base_grid_H, base_grid_W))
        selected_msk_scale_2_per_img.append(grouped_mask)

        # --- Scale 1 mask selection ---
        grouped_mask_large = F.interpolate(grouped_mask.float().unsqueeze(0).unsqueeze(0),
                                           size=(base_grid_H*2, base_grid_W*2),
                                           mode='nearest').squeeze(0).squeeze(0).bool()
        mask_scale_1 = torch.logical_not(grouped_mask_large)
        flat_mask = mask_scale_1.view(-1)
        num_true = flat_mask.sum().item()
        if num_true > num_scale_1:
            # Set excess True to False (keep the first num_scale_1 True indices)
            true_indices = flat_mask.nonzero(as_tuple=True)[0]
            flat_mask[true_indices[num_scale_1:]] = False
        elif num_true < num_scale_1:
            # Set extra False to True (fill up to num_scale_1)
            false_indices = (~flat_mask).nonzero(as_tuple=True)[0]
            flat_mask[false_indices[:(num_scale_1 - num_true)]] = True
        mask_scale_1 = flat_mask.view(mask_scale_1.shape)
        selected_msk_scale_1_per_img.append(mask_scale_1)

    selected_msk_scale_1 = torch.stack(selected_msk_scale_1_per_img, dim=0)
    selected_msk_scale_2 = torch.stack(selected_msk_scale_2_per_img, dim=0)

    # Per-image assertions for robustness
    per_img_sum_1 = selected_msk_scale_1.view(B, -1).sum(dim=1)
    per_img_sum_2 = selected_msk_scale_2.view(B, -1).sum(dim=1)
    if not torch.all(per_img_sum_1 == num_scale_1):
        raise ValueError(f"Each image in selected_msk_scale_1 must have exactly {num_scale_1} True values, got: {per_img_sum_1}")
    if not torch.all(per_img_sum_2 == num_scale_2):
        raise ValueError(f"Each image in selected_msk_scale_2 must have exactly {num_scale_2} True values, got: {per_img_sum_2}")

    return selected_msk_scale_1, selected_msk_scale_2

def policy_indices_no_sharing(images, patch_size):
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size
    selected_msk_scale_2 = torch.zeros((B, base_grid_H // 2, base_grid_W // 2), dtype=torch.bool)
    selected_msk_scale_1 = torch.ones((B, base_grid_H, base_grid_W), dtype=torch.bool)
    return (selected_msk_scale_1, selected_msk_scale_2)

def images_to_patches(images, patch_size, policy_indices):
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size
    patch_scale_1 = rearrange(images, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H, gw=base_grid_W)
    scale_value_1 = torch.ones([B, base_grid_H, base_grid_W, 1], device=images.device)
    patch_code_scale_1 = torch.cat([
        scale_value_1,
        torch.linspace(0, base_grid_H-1, base_grid_H, device=images.device).view(-1, 1, 1).expand_as(scale_value_1),
        torch.linspace(0, base_grid_W-1, base_grid_W, device=images.device).view(1, -1, 1).expand_as(scale_value_1)
    ], dim=3)
    patch_scale_2 = rearrange(F.interpolate(images, scale_factor=0.5, mode='bilinear',
                                            align_corners=False, recompute_scale_factor=False),
                              'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H//2, gw=base_grid_W//2)
    patch_code_scale_2 = torch.clone(patch_code_scale_1)[:, ::2, ::2, :]
    patch_code_scale_2[:, :, :, 0] = 2
    (selected_msk_scale_1, selected_msk_scale_2) = policy_indices
    # Before indexing, ensure selected_msk_scale_1 is on the same device as patch_code_scale_1
    selected_msk_scale_1 = selected_msk_scale_1.to(patch_code_scale_1.device)
    patch_code_scale_1_selected = patch_code_scale_1[selected_msk_scale_1]
    # Ensure selected_msk_scale_2 is on the same device as patch_code_scale_2
    selected_msk_scale_2 = selected_msk_scale_2.to(patch_code_scale_2.device)
    patch_code_scale_2_selected = patch_code_scale_2[selected_msk_scale_2]
    patch_code_scale_2_selected = rearrange(patch_code_scale_2_selected, '(b np) c -> b np c', b=B)
    patch_scale_2_selected = patch_scale_2[selected_msk_scale_2]
    patch_scale_2_selected = rearrange(patch_scale_2_selected, '(b np) c h w -> b np c h w', b=B)
    patch_code_scale_1_selected = patch_code_scale_1_selected
    patch_code_scale_1_selected = rearrange(patch_code_scale_1_selected, '(b np) c -> b np c', b=B)
    patch_scale_1_selected = patch_scale_1[selected_msk_scale_1]
    patch_scale_1_selected = rearrange(patch_scale_1_selected, '(b np) c h w -> b np c h w', b=B)
    patches_total = torch.cat([patch_scale_1_selected, patch_scale_2_selected], dim=1)
    patch_code_total = torch.cat([patch_code_scale_1_selected, patch_code_scale_2_selected], dim=1)
    patches_total = rearrange(patches_total, 'b np c ps_h ps_w -> b c ps_h (np ps_w)')
    return patches_total, patch_code_total

def rearrange(tensor, pattern, **axes_lengths):
    # Simple wrapper for einops.rearrange
    from einops import rearrange as einops_rearrange
    return einops_rearrange(tensor, pattern, **axes_lengths)

@MODELS.register_module()
class CTSVisionTransformer(VisionTransformer):
    def __init__(self, policy_method='policy_net', policy_schedule=(612, 103), policynet_ckpt=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_method = policy_method
        self.policy_schedule = policy_schedule
        self.policynet_ckpt = policynet_ckpt
        if self.policy_method == 'policy_net':
            self.policy_net = PolicyNet()
            if self.policynet_ckpt is not None:
                state = torch.load(self.policynet_ckpt, map_location='cpu')
                missing_keys, unexpected_keys = self.policy_net.load_state_dict(state, strict=False)
                if not missing_keys:
                    print("[CTSVisionTransformer] PolicyNet weights loaded successfully.")
                else:
                    print(f"[CTSVisionTransformer] PolicyNet loaded with missing keys: {missing_keys}")
            self.policy_net.eval()
            for param in self.policy_net.parameters():
                param.requires_grad = False

    def tokens_to_map(self, tokens, policy_code, H, W):
        B, N, C = tokens.shape
        feat_map = tokens.new_zeros((B, C, H, W))
        for b in range(B):
            hs = policy_code[b, :, 1].long()
            ws = policy_code[b, :, 2].long()
            feat_map[b, :, hs, ws] = tokens[b].transpose(0, 1)
        return feat_map

    def forward(self, x):
        B, C, H, W = x.shape
        PS = self.patch_size
        grid_H, grid_W = self.img_size[0] // PS, self.img_size[1] // PS
        if self.policy_method == 'policy_net':
            with torch.no_grad():
                policynet_pred = self.policy_net(x)
            policy_indices = policy_indices_by_policynet_pred(
                x, patch_size=PS, policy_schedule=self.policy_schedule, 
                policynet_pred=policynet_pred
            )
        elif self.policy_method == 'no_sharing':
            policy_indices = policy_indices_no_sharing(x, patch_size=PS)
        else:
            raise NotImplementedError(f'Policy method {self.policy_method} not supported')
        x_patches, policy_code = images_to_patches(x, patch_size=PS, policy_indices=policy_indices)
        num_patches = policy_code.size(1)
        # Patch embedding
        x_emb = self.patch_embed(x_patches)[0]  # [B, N, C], ignore hw_shape
        # Select positional embeddings for selected tokens
        pos_embed_full = self.pos_embed[:, 1:, :] if self.with_cls_token else self.pos_embed
        pos_embed_full = pos_embed_full.view(1, grid_H, grid_W, -1)
        hs = policy_code[:, :, 1].long()
        ws = policy_code[:, :, 2].long()
        pos_embed_selected = []
        for b in range(B):
            pos_embed_selected.append(pos_embed_full[0, hs[b], ws[b], :])
        pos_embed_selected = torch.stack(pos_embed_selected, dim=0)  # [B, N, C]
        x = x_emb + pos_embed_selected
        # Add class token if needed
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        # Forward through transformer layers
        outs = []
        hw_shape = (grid_H, grid_W)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and getattr(self, 'final_norm', False):
                x = self.norm1(x)
            if i in self.out_indices:
                # Remove class token if present
                if self.with_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                # Scatter tokens back to spatial map using policy_code
                out_feat = self.tokens_to_map(out, policy_code, grid_H, grid_W)
                outs.append(out_feat)
        return tuple(outs) 