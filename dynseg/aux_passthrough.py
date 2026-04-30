from typing import Tuple

import torch
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from torch import Tensor, nn


@MODELS.register_module()
class AuxPassthrough(nn.Module):
    """
    Auxiliary head for dynamic segmentation.

    This module acts as a pass-through for segmentation logits, enabling loss computation
    using provided loss functions without altering the logits.

    Input:
        - inputs (Tuple[Tensor]): Segmentation logits from the backbone or decode head.
        - batch_data_samples (SampleList): List of data samples containing ground truth segmentation maps.

    Output:
        - Dictionary containing computed loss values based on provided loss functions.
    """

    def __init__(self, loss_decode=None, **kwargs):
        super(AuxPassthrough, self).__init__(**kwargs)

        # Build loss functions from configuration
        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(
                f"loss_decode must be a dict or sequence of dict, but got {type(loss_decode)}"
            )

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """
        Stack ground truth segmentation maps from all samples in the batch.

        Args:
            batch_data_samples (SampleList): List of segmentation data samples.

        Returns:
            Tensor: Batched ground truth segmentation maps of shape (N, H, W).
        """
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0).squeeze(1)

    def loss_by_feat(self, seg_logits: Tensor, batch_data_samples: SampleList) -> dict:
        """
        Compute segmentation loss using provided logits and data samples.

        Args:
            seg_logits (Tensor): Segmentation logits.
            batch_data_samples (SampleList): List of segmentation data samples.

        Returns:
            dict[str, Tensor]: Dictionary of computed loss values.
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        losses_decode = (
            [self.loss_decode]
            if not isinstance(self.loss_decode, nn.ModuleList)
            else self.loss_decode
        )

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits, seg_label, weight=None
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits, seg_label, weight=None
                )

        return loss

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        """
        Pass-through forward function.

        Args:
            inputs (Tuple[Tensor]): Input logits.

        Returns:
            Tensor: Unchanged input logits.
        """
        return inputs

    def loss(
        self,
        inputs: Tuple[Tensor],
        batch_data_samples: SampleList,
        train_cfg: ConfigType,
    ) -> dict:
        """
        Forward function for training.

        Args:
            inputs (Tuple[Tensor]): Input segmentation logits.
            batch_data_samples (SampleList): List of segmentation data samples.
            train_cfg (ConfigType): Training configuration dictionary.

        Returns:
            dict[str, Tensor]: Dictionary of computed loss values.
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
