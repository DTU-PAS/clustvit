import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops


def get_clusters(masks: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Divide each mask in the batch into non-overlapping patches.
    For each patch:
      - If all pixels are identical, assign that value to the patch.
      - Otherwise, assign -1 to indicate mixed values.

    Args:
        masks (torch.Tensor): (B, H, W) tensor of integer masks.
        patch_size (int): Size of the square patches.

    Returns:
        torch.Tensor: (B, H_patch, W_patch) tensor with patch values,
                      where H_patch = H // patch_size, W_patch = W // patch_size.
    """
    batch_size, H, W = masks.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Split masks into patches of shape (B, H_patch, W_patch, patch_size, patch_size)
    patches = masks.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, num_patches_h, num_patches_w, -1)

    patch_max = patches.max(dim=-1)[0]
    patch_min = patches.min(dim=-1)[0]

    homogeneous = patch_max == patch_min
    result = torch.where(
        homogeneous,
        patch_max,
        torch.tensor(-1, device=masks.device, dtype=masks.dtype),
    )
    return result


def filter_topk_clusters(patch_values: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each sample, keep only the top-k most frequent cluster values.
    All other values are set to -1.
    Then remap the top-k cluster values to a consecutive range [0, k-1].

    Args:
        patch_values (torch.Tensor): (B, H_patch, W_patch) tensor of cluster IDs.
        k (int): Number of top clusters to retain per sample.

    Returns:
        torch.Tensor: (B, H_patch, W_patch) tensor with remapped cluster IDs,
                      other values set to -1.
    """
    batch_size = patch_values.size(0)
    new_patch_values = torch.full_like(patch_values, -1)

    for i in range(batch_size):
        sample = patch_values[i]
        flat = sample.view(-1)
        valid = flat[flat != -1]

        if valid.numel() == 0:
            continue

        unique, counts = torch.unique(valid, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)
        topk_clusters = unique[sorted_indices][:k]

        mapping = {
            int(val.item()): new_idx for new_idx, val in enumerate(topk_clusters)
        }

        for old_val, new_val in mapping.items():
            new_patch_values[i][sample == old_val] = new_val

    return new_patch_values


def refine_masks(
    mask_batch: torch.Tensor,
    min_area: int = 50,
    border_value: int = 0,
) -> torch.Tensor:
    """
    Post-process a batch of integer masks by removing small connected components
    and eroding remaining blobs by 1 pixel. Adds a 1-pixel border of `border_value`.

    Steps:
      1) Add 1-pixel border with `border_value`.
      2) Remove connected components smaller than `min_area`.
      3) Erode blobs by 1 pixel to shrink boundaries.

    Args:
        mask_batch (torch.Tensor): (B, H, W) integer masks (0=background, >0=clusters).
        min_area (int): Minimum pixel area for blobs to keep.
        border_value (int): Value to assign to 1-pixel border and background.

    Returns:
        torch.Tensor: (B, H, W) post-processed masks.
    """
    device = mask_batch.device
    dtype = mask_batch.dtype
    out = torch.zeros_like(mask_batch)

    for i in range(mask_batch.size(0)):
        m = mask_batch[i].cpu().numpy().astype(np.int32)

        # Add 1-pixel border
        m[0, :] = border_value
        m[-1, :] = border_value
        m[:, 0] = border_value
        m[:, -1] = border_value

        # Remove small blobs by label
        lbl = label(m, background=border_value)
        for region in regionprops(lbl):
            if region.area < min_area:
                m[lbl == region.label] = border_value

        # Erode blobs by 1 pixel
        binary = (m != border_value).astype(np.uint8)
        eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
        m[eroded == 0] = border_value

        out[i] = torch.from_numpy(m).to(device).type(dtype)

    return out


def process_masks(
    masks: torch.Tensor,
    patch_size: int = 16,
    k: int = 3,
) -> torch.Tensor:
    """
    Pipeline to process batch of masks into top-k clusters per patch.

    Steps:
      1) Extract homogeneous patches via `get_clusters`.
      2) Replace 255 by -1 (invalid).
      3) Keep only top-k clusters with `filter_topk_clusters`.
      4) Optionally refine masks (currently commented out).
      5) Shift cluster indices by +1.

    Args:
        masks (torch.Tensor): (B, H, W) input masks.
        patch_size (int): Patch size.
        k (int): Number of top clusters to keep.

    Returns:
        torch.Tensor: (B, H_patch, W_patch) processed cluster mask.
    """
    clusters = get_clusters(masks, patch_size)
    clusters[clusters == 255] = -1

    pseudo_clusters = filter_topk_clusters(clusters, k)

    # Uncomment to apply mask refinement:
    # pseudo_clusters = refine_masks(pseudo_clusters, min_area=50, border_value=-1)

    pseudo_clusters = pseudo_clusters + 1
    return pseudo_clusters


def process_mask_class(
    masks: torch.Tensor,
    patch_size: int = 16,
    k: int = 3,
) -> torch.Tensor:
    """
    Simplified mask processing to obtain patch clusters,
    with no top-k filtering or refinement.

    Args:
        masks (torch.Tensor): (B, H, W) input masks.
        patch_size (int): Patch size.
        k (int): Number of top clusters (unused here).

    Returns:
        torch.Tensor: (B, H_patch, W_patch) patch clusters with -1 and 255 mapped to 0.
    """
    clusters = get_clusters(masks, patch_size)
    clusters[clusters == -1] = 0
    clusters[clusters == 255] = 0
    return clusters
