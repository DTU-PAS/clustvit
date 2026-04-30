import logging
from typing import List, Optional

import torch.nn.functional as F
from mmengine.logging import print_log
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import (
    ConfigType,
    ForwardResults,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
)
from torch import Tensor


@MODELS.register_module()
class DynSegEncoderDecoder(EncoderDecoder):
    """
    Dynamic Segmentation Encoder-Decoder model extending the base EncoderDecoder.

    Supports dynamic token clustering in the backbone and returns cluster logits
    alongside segmentation logits.

    Args:
        backbone (ConfigType): Backbone config.
        decode_head (ConfigType): Decode head config.
        neck (OptConfigType, optional): Neck config.
        auxiliary_head (OptConfigType, optional): Auxiliary head config.
        train_cfg (OptConfigType, optional): Training config.
        test_cfg (OptConfigType, optional): Testing config.
        data_preprocessor (OptConfigType, optional): Data preprocessor config.
        pretrained (Optional[str], optional): Path to pretrained weights.
        init_cfg (OptMultiConfig, optional): Initialization config.
    """

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """
        Extract features from input images using the backbone.

        Args:
            inputs (Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            List[Tensor]: A list containing features from backbone and cluster logits.
        """
        x, logits = self.backbone(inputs)
        # Optional: pass features through neck if implemented
        # if self.with_neck:
        #     x = self.neck(x)

        return x, logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x, logits = self.extract_feat(inputs)
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        # Auxiliary head loss if available
        if self.with_auxiliary_head:
            aux_heads = self.auxiliary_head if isinstance(self.auxiliary_head, (list, tuple)) else [self.auxiliary_head]
            for idx, aux_head in enumerate(aux_heads):
                # Pass logits to AuxPassthrough, features to ATMHeadV2 (using in_index if present)
                if aux_head.__class__.__name__ == "AuxPassthrough":
                    aux_input = logits
                else:
                    # Default: pass features
                    aux_input = x
                loss_aux = aux_head.loss(aux_input, data_samples, self.train_cfg)
                losses.update({f'aux_{idx}_{k}': v for k, v in loss_aux.items()})

        return losses

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """
        Forward inputs through backbone and decode to semantic segmentation logits.

        Args:
            inputs (Tensor): Input images tensor of shape (B, C, H, W).
            batch_img_metas (List[dict]): List of image metadata dictionaries.

        Returns:
            Tensor: Segmentation logits tensor.
            Tensor: Cluster logits tensor.
        """
        x, clust_logits = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

        return seg_logits, clust_logits

    def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """
        Perform inference using sliding window with overlap for large images.

        Args:
            inputs (Tensor): Input images tensor of shape (N, C, H, W).
            batch_img_metas (List[dict]): List of image metadata.

        Returns:
            Tensor: Segmentation logits of shape (N, num_classes, H, W).
            None: Cluster logits not returned in sliding inference.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]["img_shape"] = crop_img.shape[2:]

                crop_seg_logit, _ = self.encode_decode(crop_img, batch_img_metas)

                preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (
            count_mat == 0
        ).sum() == 0, "Some pixels were not covered by sliding windows"
        seg_logits = preds / count_mat

        return seg_logits, None

    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """
        Perform inference on whole images.

        Args:
            inputs (Tensor): Input images tensor of shape (N, C, H, W).
            batch_img_metas (List[dict]): List of image metadata.

        Returns:
            Tensor: Segmentation logits.
            Tensor: Cluster logits.
        """
        seg_logits, clust_logits = self.encode_decode(inputs, batch_img_metas)
        return seg_logits, clust_logits

    def inference(
        self, inputs: Tensor, batch_img_metas: List[dict], return_logits=False
    ) -> Tensor:
        """
        Inference method supporting sliding-window and whole-image modes.

        Args:
            inputs (Tensor): Input images tensor of shape (N, C, H, W).
            batch_img_metas (List[dict]): List of image metadata.
            return_logits (bool): Whether to return logits (segmentation and cluster).

        Returns:
            Tensor or tuple: Segmentation logits or tuple of (segmentation logits, cluster logits).
        """
        mode = self.test_cfg.get("mode", "whole")
        assert mode in [
            "slide",
            "whole",
        ], f'Only "slide" or "whole" test mode are supported, but got "{mode}".'

        ori_shape = batch_img_metas[0]["ori_shape"]
        if not all(meta["ori_shape"] == ori_shape for meta in batch_img_metas):
            print_log(
                "Image shapes are different in the batch.",
                logger="current",
                level=logging.WARN,
            )

        if mode == "slide":
            seg_logit, clust_logits = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit, clust_logits = self.whole_inference(inputs, batch_img_metas)

        if return_logits:
            return seg_logit, clust_logits

        return seg_logit

    def predict(
        self, inputs: Tensor, data_samples: OptSampleList = None, return_logits=False
    ) -> SampleList:
        """
        Predict segmentation results from inputs with optional post-processing.

        Args:
            inputs (Tensor): Input tensor of shape (N, C, H, W).
            data_samples (List[SegDataSample], optional): Data samples with metadata.

        Returns:
            List[SegDataSample]: List of segmentation data samples with predictions.
        """
        if data_samples is not None:
            batch_img_metas = [sample.metainfo for sample in data_samples]
        else:
            # Default meta info if data_samples not provided
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        if return_logits:
            seg_logits, clust_logits = self.inference(
                inputs, batch_img_metas, return_logits=True
            )
            return self.postprocess_result(seg_logits, data_samples), clust_logits

        seg_logits = self.inference(inputs, batch_img_metas, return_logits=False)
        return self.postprocess_result(seg_logits, data_samples)

    def forward(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
        return_logits: bool = False,
    ) -> ForwardResults:
        """
        Unified forward method handling training, prediction, and loss computation.

        Args:
            inputs (Tensor): Input tensor.
            data_samples (List[SegDataSample], optional): Data samples with labels.
            mode (str): Mode of operation, one of 'tensor', 'predict', or 'loss'.
            return_logits (bool): Whether to return logits in prediction.

        Returns:
            Depending on mode:
            - 'tensor': Tensor or tuple of tensors.
            - 'predict': List of SegDataSample with predictions.
            - 'loss': Dict of loss tensors.
        """
        if mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples, return_logits=return_logits)
        elif mode == "tensor":
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". Only supports "loss", "predict", and "tensor".'
            )
