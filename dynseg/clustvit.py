# Copyright (c) OpenMMLab. All rights reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.model import ModuleList
from mmseg.models.backbones.vit import TransformerEncoderLayer, VisionTransformer
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import print_log
import math
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from torch.nn.modules.batchnorm import _BatchNorm                                       

def remap_vit_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remap attention qkv/proj to in_proj/out_proj
        if '.attn.qkv.weight' in k:
            new_k = k.replace('.attn.qkv.weight', '.attn.attn.in_proj_weight')
        elif '.attn.qkv.bias' in k:
            new_k = k.replace('.attn.qkv.bias', '.attn.attn.in_proj_bias')
        elif '.attn.proj.weight' in k:
            new_k = k.replace('.attn.proj.weight', '.attn.attn.out_proj.weight')
        elif '.attn.proj.bias' in k:
            new_k = k.replace('.attn.proj.bias', '.attn.attn.out_proj.bias')
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict
        

class Clusterer(nn.Module):
    """
    A simple feed-forward neural network for clustering input features
    into a fixed number of output classes.

    Architecture:
        Input Linear Layer -> ReLU Activation -> Output Linear Layer

    Args:
        input_dim (int): Dimension of input feature vectors.
        hidden_dim (int, optional): Number of hidden units. Default: 64.
        output_dim (int, optional): Number of output classes/clusters. Default: 4.
        init_std (float, optional): Standard deviation for weight initialization.
                                    Used only if `init_zero` is True. Default: 1e-2.
        init_zero (bool, optional): If True, initializes weights with small random
                                    values and biases to zero. Default: False.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

    Outputs:
        torch.Tensor: Output tensor of shape (batch_size, output_dim),
                      representing logits or cluster scores.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 4,
        init_std: float = 1e-2,
        init_zero: bool = False,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        if init_zero:
            self.initialize_weights_small(init_std)

    def initialize_weights_small(self, init_std: float) -> None:
        """
        Initialize linear layers' weights with small random normal values
        and biases with zeros.

        Args:
            init_std (float): Standard deviation for weight initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the clusterer network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        return self.net(x)


class ClusterViTEncoderLayer(TransformerEncoderLayer):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        batch_first=True,
        attn_cfg=dict(),
        ffn_cfg=dict(),
        with_cp=False,
    ):

        super(ClusterViTEncoderLayer, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            batch_first=batch_first,
            with_cp=with_cp,
        )
        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=(
                    dict(type="DropPath", drop_prob=drop_path_rate)
                    if drop_path_rate > 0
                    else None
                ),
                act_cfg=act_cfg,
                add_identity=False,  # important, because we do this manually
            )
        )
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def forward(
        self,
        x,
        key_padding_mask=None,
    ):

        def _inner_forward(x, key_padding_mask=None):
            x = self.attn(self.norm1(x), identity=x, key_padding_mask=key_padding_mask)
            x_shift = self.ffn(self.norm2(x), identity=None)

            return x + x_shift  # done manually, out of experimentation convenience

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, key_padding_mask=key_padding_mask)
        else:
            x = _inner_forward(x, key_padding_mask=key_padding_mask)
        return x


@MODELS.register_module()
class ClustViT(VisionTransformer):
    """Wrapper of Vision Transformer for Clust ViT. The arguments stay the same as MMSeg.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        patch_pad  (str | int | None): The padding method in patch embedding.
            Default: 'corner'.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_origin (bool): Whether to output the original input embedding.
            Default: False
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_bias (dict): Whether use bias in convolution of PatchEmbed Block.
            Default: True.
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        pre_norm (bool): Whether to add a norm before Transformer Layers.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        frozen_exclude (List): List of parameters that are not to be frozen.
            Default: ["all"], "all" means there are no frozen parameters.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        patch_pad="corner",
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_origin=False,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type="LN"),
        act_cfg=dict(type="GELU"),
        patch_norm=False,
        patch_bias=False,
        pre_norm=False,
        final_norm=False,
        interpolate_mode="bicubic",
        num_fcs=2,
        norm_eval=False,
        with_cp=False,
        frozen_exclude=["all"],
        pretrained=None,
        init_cfg=None,
        num_clusters=3,
        injection_points=[4],
        cluster_mlp_size=3072,
        init_weights_zero=False,
    ):
        super(ClustViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            patch_pad=patch_pad,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_origin=out_origin,
            out_indices=out_indices,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            with_cls_token=with_cls_token,
            output_cls_token=output_cls_token,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            patch_norm=patch_norm,
            patch_bias=patch_bias,
            pre_norm=pre_norm,
            final_norm=final_norm,
            interpolate_mode=interpolate_mode,
            num_fcs=num_fcs,
            norm_eval=norm_eval,
            with_cp=with_cp,
            frozen_exclude=frozen_exclude,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        assert (
            len(injection_points) == 1
        ), f"Only one injection point is supported at the moment, got {len(injection_points)}"
        self.injection_point = injection_points[0]

        self.num_cluster = num_clusters
        self.clusterer = Clusterer(
            embed_dims,
            output_dim=self.num_cluster + 1,
            hidden_dim=cluster_mlp_size,
            init_zero=init_weights_zero,
        )
        self.expand_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        self.layers = ModuleList()
        for i in range(injection_points[0]):
            self.layers.append(
                ClusterViTEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True,
                )
            )

        for i in range(injection_points[0], num_layers):
            self.layers.append(
                ClusterViTEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True,
                )
            )
    

    def init_weights(self):
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if self.init_cfg.get('type') == 'Pretrained':
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'backbone.' prefix if present
                state_dict = {k[len('backbone.'):] if k.startswith('backbone.') else k: v for k, v in state_dict.items()}

                # Only remap if 'deit' is in the checkpoint path from init_cfg
                if hasattr(self, 'init_cfg') and self.init_cfg and 'checkpoint' in self.init_cfg and ('deit' in self.init_cfg['checkpoint'].lower() or 'mae' in self.init_cfg['checkpoint'].lower()):
                    state_dict = remap_vit_keys(state_dict)

            elif self.init_cfg.get('type') == 'Pretrained_Part':
                state_dict = checkpoint.copy()
                para_prefix = 'image_encoder'
                prefix_len = len(para_prefix) + 1
                for k, v in checkpoint.items():
                    state_dict.pop(k)
                    if para_prefix in k:
                        state_dict[k[prefix_len:]] = v

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _compact_tokens(self, X, mask):
        """
        Compact tokens from X (shape: [B, S, D]) using boolean mask (shape: [B, S]),
        generating a tensor of shape [B, max_valid, D] and returning also the valid token ordering.
        The boolean mask has True if a token has to be kept, False if it has to be discarded.

        Returns:
        compact_X: tensor of shape (B, max_valid, D)
        new_padding_mask: tensor of shape (B, max_valid) for attention, where is False if the token is valid,
                          True if the token is padding
        cum_indices: tensor of shape (B, S) that gives, for each valid token,
                        its index in the compact tensor (for later use)
        """
        B, S, D = X.shape
        valid_counts = mask.sum(dim=1)  # (B,)
        max_valid = int(valid_counts.max().item())  # maximum valid tokens in any batch

        # Preallocate compact tensor
        compact_X = torch.zeros(B, max_valid, D, device=X.device, dtype=X.dtype)

        # Compute cumulative indices over the sequence dimension.
        cum_indices = (mask.cumsum(dim=1) - 1).clamp(min=0)  # (B, S)

        # Get indices of valid tokens
        b_idx, s_idx = torch.nonzero(mask, as_tuple=True)
        tgt_idx = cum_indices[b_idx, s_idx].long()

        # Scatter valid tokens into the compact tensor
        compact_X.index_put_((b_idx, tgt_idx), X[b_idx, s_idx])

        # Build the new key padding mask for the compact tensor:
        token_pos = (
            torch.arange(max_valid, device=X.device, dtype=X.dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )
        new_padding_mask = token_pos >= valid_counts.unsqueeze(1)

        return compact_X, new_padding_mask, cum_indices

    def _scatter_back(self, compact_X, mask, cum_indices, orig_shape):
        """
        Scatter values from the compact tensor back into a full tensor of shape orig_shape.

        For every valid token (mask True), use cum_indices to recover the position in compact_X.

        Args:
        compact_X (torch.Tensor): tensor with compacted token representations (B, max_valid, D)
        mask (torch.BoolTensor): original mask of shape (B, S)
        cum_indices (torch.Tensor): cumulative indices computed during compaction (B, S)
        orig_shape (tuple): shape of the original X (B, S, D)

        Returns:
        full_X (torch.Tensor): tensor of shape orig_shape with values placed in original positions.
        """
        full_X = torch.zeros(orig_shape, device=compact_X.device, dtype=compact_X.dtype)
        b_idx, s_idx = torch.nonzero(mask, as_tuple=True)
        # For each valid token in the full representation, its index in compact_X is given by cum_indices.
        full_X[b_idx, s_idx] = compact_X[b_idx, cum_indices[b_idx, s_idx].long()]
        return full_X

    def _reduce_tokens(self, x):
        device = x.device  # cache device to avoid repeated .to(device)
        # split cls and points
        cls, points = x[:, 0, :], x[:, 1:, :]

        # save shapes which will be useful later
        B, N, C = points.shape

        # compute the clusters by doing the softmax (here gumbel for better exploration, may be changed later if we use soft logits)
        clust_logits = self.clusterer(points)

        all_clusters_ids = torch.argmax(clust_logits, dim=-1)
        all_clusters_ids_unique = self._count_unique(all_clusters_ids, device)

        if all_clusters_ids_unique.numel() > 1:

            # Compute the cluster mean for each cluster
            elective_points = torch.zeros(
                B, self.num_cluster + 1, C, device=device
            )  # The number of representative tokens is always fixed, but maybe it shouldn't be so
            elective_points.scatter_reduce_(
                dim=1,
                index=all_clusters_ids.unsqueeze(-1).expand(-1, -1, C),
                src=points,
                reduce="mean",
                include_self=False,
            )

            elective_points = elective_points[
                :, 1:, :
            ]  # Remember that there is also the 0 elective point, which is to be discarded
            # maybe there's a more efficient way to do this without the zero? but I don't see it now

            # take the mask of all the points that are not in the clusters
            belongs_to_cluster_mask = (
                all_clusters_ids != 0
            )  # False if the point is not in cluster, True if it it in cluster

            compact_points, new_padding_mask, cum_indices = self._compact_tokens(
                points,
                ~belongs_to_cluster_mask,  # switch to 1 for points that are not in a cluster (to be kept)
            )

            # get the input and the mask of the input
            global_x = torch.cat(
                [cls.unsqueeze(1), compact_points, elective_points], dim=1
            )
            global_x_mask = torch.cat(
                [
                    torch.zeros((B, 1), dtype=torch.int8, device=device),
                    new_padding_mask,
                    torch.zeros(B, self.num_cluster, dtype=torch.int8, device=device),
                ],
                dim=1,
            ).bool()  # 1 if the elements have to be masked

        else:
            global_x = x
            global_x_mask = None
            cum_indices = None

        return (
            global_x,
            global_x_mask,
            clust_logits,
            cum_indices,
            all_clusters_ids,
        )

    def _count_unique(self, all_clusters_ids, device):
        """Count unique values in x and return their indices."""
        # Count unique ids
        counts = torch.zeros(self.num_cluster + 1, dtype=torch.int64, device=device)
        counts.scatter_add_(
            0,
            all_clusters_ids.flatten(),
            torch.ones_like(all_clusters_ids.flatten(), device=device),
        )
        return torch.nonzero(counts > 0).squeeze()

    def _expand_tokens(self, x, orig_x, cum_indices, all_clusters_ids):
        B, N, C = x.shape
        device = x.device

        # -- split back into cls, global tokens, elective points --
        cls = x[:, :1, :]
        global_output = x[:, 1 : -self.num_cluster, :]
        elective_points = x[:, -self.num_cluster :, :]

        # mask of tokens that were pruned (i.e. we need to restore them)
        belongs_to_cluster_mask = all_clusters_ids != 0  # True=clustered, False=pruned

        # scatter global_output back into full sequence
        full_attn_output = self._scatter_back(
            global_output, ~belongs_to_cluster_mask, cum_indices, orig_x.shape
        )

        # now we have full_attn_output for all positions,
        # but it’s not “refined” yet for those pruned positions.
        # We’ll only run the MLP on those positions:
        b_idx, s_idx = torch.nonzero(belongs_to_cluster_mask, as_tuple=True)
        # gather the originals and the outputs
        orig_feats = orig_x[b_idx, s_idx]  # (num_pruned, C)
        only_cluster_ids = all_clusters_ids[b_idx, s_idx] - 1  # (num_pruned,)
        # updated_feats = full_attn_output[b_idx, s_idx]  # (num_pruned, C)
        # distributed_elective_points = torch.gather(elective_points, 1, only_cluster_ids)
        expanded_elective_points = elective_points[b_idx, only_cluster_ids]

        # concatenate and run through MLP
        mlp_in = torch.cat(
            [orig_feats, expanded_elective_points], dim=-1
        )  # (num_pruned, 2C)
        refined = self.expand_mlp(mlp_in)  # (num_pruned, C)

        # write back only those positions
        full_attn_output[b_idx, s_idx] = refined

        out = torch.cat([cls, full_attn_output], dim=1)
        return out

    def forward(self, inputs):
        # # Decide between training and optimized inference
        # if not self.training:
        #     return self._inference_forward(inputs)

        # PREPREOCESSING
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if self.pre_norm:
            x = self.pre_ln(x)

        outs = []
        if self.out_origin:
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = x[:, 1:]
            else:
                out = x
            B, _, C = out.shape
            out = (
                out.reshape(B, hw_shape[0], hw_shape[1], C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if self.output_cls_token:
                out = [out, x[:, 0]]
            outs.append(out)

        # ENCODER
        x_mask = None
        original_x = None
        cum_indices = None  # Initialize to avoid reference before assignment
        for i, layer in enumerate(self.layers):

            if i == self.injection_point:
                original_x = x[:, 1:, :]  # x[:, 1:, :].clone()
                x, x_mask, clust_logits, cum_indices, all_clusters_ids = (
                    self._reduce_tokens(x)
                )

            x = layer(x, key_padding_mask=x_mask)
            if i == len(self.layers) - 1:
                # rebuild together the tokens
                if (
                    cum_indices is not None
                ):  # Indicates if there has been clusterization or not
                    x = self._expand_tokens(
                        x,
                        original_x,
                        cum_indices,
                        all_clusters_ids,
                    )
                    cum_indices = None  # reset to avoid confusion
                if self.final_norm:
                    x = self.norm1(x)

            if i in self.out_indices:
                if (
                    cum_indices is not None
                ):  # Indicates if there has been clusterization or not
                    out = self._expand_tokens(
                        x,
                        original_x,
                        cum_indices,
                        all_clusters_ids,
                    )[:, 1:]
                else:
                    out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs), clust_logits.unsqueeze(0)
