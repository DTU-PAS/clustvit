import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses.utils import weighted_loss
from mmseg.registry import MODELS
from scipy.optimize import linear_sum_assignment

from .utils import process_masks


@weighted_loss
def clust_loss(
    pred,
    target,
    weight=None,
    reduction="mean",
    avg_factor=None,
    num_clusters=3,
    ignore_index=None,
):
    """
    Cross-entropy clustering loss computed over multiple prediction layers.

    Args:
        pred (list[torch.Tensor]): List of layer outputs, each [B, N, C].
        target (torch.Tensor): Ground truth masks.
        weight (torch.Tensor or None): Sample weights.
        reduction (str): Reduction mode: 'none', 'mean', or 'sum'.
        avg_factor (int or None): Normalization factor.
        num_clusters (int): Number of clusters for pseudo labels.
        ignore_index (int or None): Label to ignore in loss computation.

    Returns:
        torch.Tensor: Aggregated cross-entropy loss over layers.
    """
    clust_loss = 0.0
    pseudo_clust_labels = process_masks(target, k=num_clusters)
    pseudo_clust_labels = pseudo_clust_labels.flatten(1)  # [B, N]

    for layer_clust in pred:
        clust_loss += F.cross_entropy(
            layer_clust.permute(0, 2, 1),
            pseudo_clust_labels.long(),
            reduction=reduction,
            ignore_index=-100 if ignore_index is None else ignore_index,
        )
    return clust_loss


@torch.no_grad()
def batch_hungarian_perms(
    pred_probs: torch.Tensor, target_labels: torch.Tensor, C: int
) -> torch.Tensor:
    """
    Compute optimal cluster label permutations per batch element using Hungarian algorithm.

    Args:
        pred_probs (torch.Tensor): Predicted probabilities, shape [B, N, C].
        target_labels (torch.Tensor): Ground truth cluster labels, shape [B, N].
        C (int): Number of clusters/classes.

    Returns:
        torch.Tensor: Batch of permutation arrays mapping predicted to true clusters, shape [B, C].
    """
    B, N, _ = pred_probs.shape

    # Get predicted cluster labels and one-hot encodings
    pred_labels = pred_probs.argmax(dim=2)  # [B, N]
    pred_1hot = F.one_hot(pred_labels, C).float()  # [B, N, C]
    true_1hot = F.one_hot(target_labels, C).float()  # [B, N, C]

    # Compute confusion matrices [B, C, C]
    confusion = torch.einsum("bnc,bnm->bcm", pred_1hot, true_1hot)

    # Convert to CPU numpy for Hungarian assignment
    cost_cpu = confusion.cpu().numpy()
    perms = []
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost_cpu[b], maximize=True)
        p = np.zeros(C, dtype=int)
        p[row_ind] = col_ind
        perms.append(p)

    perms = torch.from_numpy(np.stack(perms)).to(pred_probs.device)  # [B, C]
    return perms


@MODELS.register_module()
class ClustLoss(nn.Module):
    """
    Clustering cross-entropy loss module.

    Args:
        loss_name (str): Name identifier for the loss.
        reduction (str): Reduction method: 'mean', 'sum', or 'none'.
        loss_weight (float): Weighting factor for the loss.
        num_clusters (int): Number of clusters for pseudo labels.
        ignore_index (int or None): Label to ignore during loss computation.
    """

    def __init__(
        self,
        loss_name="clust_ce",
        reduction="mean",
        loss_weight=1.0,
        num_clusters=3,
        ignore_index=None,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.num_clusters = num_clusters
        self.ignore_index = ignore_index

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """
        Forward pass for clustering cross-entropy loss.

        Args:
            pred (list[torch.Tensor]): List of predictions [B, N, C].
            target (torch.Tensor): Ground truth masks.
            weight (torch.Tensor or None): Sample weights.
            avg_factor (int or None): Normalization factor.
            reduction_override (str or None): Override reduction method.

        Returns:
            torch.Tensor: Computed loss value.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        loss = self.loss_weight * clust_loss(
            pred,
            target,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            num_clusters=self.num_clusters,
            ignore_index=self.ignore_index,
        )
        return loss


