# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmseg.registry import MODELS
from PIL import Image
from tqdm import tqdm

import dynseg

# import loss
from dynseg.loss.utils import process_masks


def colorize_preds(preds, class_colors):
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1).type(torch.int8)
    preds[preds == 255] = 150
    class_colors = torch.tensor(class_colors, dtype=torch.uint8).cpu()
    preds_color = class_colors[preds.detach().cpu()]
    preds_color = preds_color.permute(0, 3, 1, 2)
    return preds_color


def colorize_class_labels(labels, class_colors):
    if labels.dim() == 4:
        labels = torch.argmax(labels, dim=1).type(torch.int8)
    class_colors = torch.tensor(class_colors, dtype=torch.uint8).cpu()
    preds_color = class_colors[labels.detach().cpu()]
    preds_color = preds_color.permute(0, 3, 1, 2)
    return preds_color


def plot_classwise_boxplots(all_freqs, save_name="classwise_boxplot.png"):
    k_max = all_freqs.shape[1] - 1
    plt.figure(figsize=(10, 6))

    plt.boxplot(
        all_freqs,
        positions=range(k_max + 1),
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
        flierprops=dict(
            markerfacecolor="red", marker="o", markersize=4, linestyle="none"
        ),
    )

    plt.title("Class-wise Frequency Distribution Over Images")
    plt.xlabel("Class k")
    plt.ylabel("Relative Frequency")
    plt.xticks(ticks=range(k_max + 1), labels=[str(i) for i in range(k_max + 1)])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def unnormalize_channel_rev(
    np_image, mean=np.array([127.5, 127.5, 127.5]), std=np.array([127.5, 127.5, 127.5])
):
    unnormalized_image = (np_image * std) + mean
    unnormalized_image = (unnormalized_image).astype(np.uint8)
    return np.clip(unnormalized_image, 0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg save cluster features")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--log-interval", type=int, default=50, help="interval of logging"
    )
    parser.add_argument(
        "--work-dir",
        help=("if specified, the results will be dumped " "into the directory as json"),
    )
    parser.add_argument("--repeat-times", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get("default_scope", "mmseg"))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.work_dir is not None:
        mkdir_or_exist(osp.abspath(args.work_dir), f"clust_vis_{timestamp}")
        work_dir = args.work_dir
    else:
        work_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(args.config))[0],
            f"clust_vis_{timestamp}",
        )
        mkdir_or_exist(osp.abspath(work_dir))

    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None

    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    cfg.model.train_cfg = None
    cfg.model.test_cfg.mode = "whole"

    if "num_clusters" in cfg.model.backbone:
        num_clusters = cfg.model.backbone.num_clusters
    else:
        num_clusters = 3

    model = MODELS.build(cfg.model)

    if "checkpoint" in args and osp.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    if torch.cuda.is_available():
        model = model.cuda()

    model = revert_sync_batchnorm(model)
    model.eval()

    total_images = 200

    for image_num, data in enumerate(data_loader):
        data = model.data_preprocessor(data, True)
        inputs = data["inputs"]
        data_samples = data["data_samples"]

        with torch.no_grad():
            out, logits = model(
                inputs, data_samples, mode="predict", return_logits=True
            )
            norm_in = unnormalize_channel_rev(inputs[0].permute(1, 2, 0).cpu().numpy())

            num_patches = (
                int(out[0].img_shape[0] / 16) + (out[0].img_shape[0] % 16 > 0),
                int(out[0].img_shape[1] / 16) + (out[0].img_shape[1] % 16 > 0),
            )

            # Ground truth and predictions
            gt = colorize_preds(out[0].gt_sem_seg.data, RUMEX_COLORS).cpu().numpy()
            gt = np.array(Image.fromarray(gt[0].transpose((1, 2, 0))))
            pred = colorize_preds(out[0].pred_sem_seg.data, RUMEX_COLORS).cpu().numpy()
            pred = np.array(Image.fromarray(pred[0].transpose((1, 2, 0))))

            # Resize everything to GT size
            gt_size = gt.shape[:2]
            norm_in = np.array(
                Image.fromarray(norm_in).resize(gt_size[::-1], Image.BILINEAR)
            )
            pred = np.array(Image.fromarray(pred).resize(gt_size[::-1], Image.NEAREST))

            mask = out[0].gt_sem_seg.data[0].cpu().numpy()

            # suppose mask is a numpy array of shape (H, W)
            H, W = mask.shape

            # compute scaling factor
            scale = 512 / max(H, W)  # largest dimension becomes 512

            new_H = int(H * scale)
            new_W = int(W * scale)

            # resize with nearest neighbor (important for masks!)
            inp = cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

            # inp = Image.fromarray(mask).resize((512, 512), Image.NEAREST)

            pseudo_labels = process_masks(
                torch.tensor(np.array(inp)).unsqueeze(0), k=num_clusters
            )


            print(pseudo_labels.shape)
            print(pseudo_labels.unique())
            gt_mask = colorize_class_labels(pseudo_labels, COLORS)
            gt_mask = gt_mask[0].cpu().numpy().transpose((1, 2, 0))
            gt_mask = np.array(
                Image.fromarray(gt_mask).resize(gt_size[::-1], Image.NEAREST)
            )

            logits_pred = torch.argmax(logits[0], dim=-1).cpu().numpy()
            mask = np.zeros((*num_patches, 3), dtype=np.uint8)
            curr_clust = logits_pred.reshape(num_patches)
            for i, c in enumerate(range(0, num_clusters + 1)):
                mask[curr_clust == c] = COLORS[i]
            mask = np.array(Image.fromarray(mask).resize(gt_size[::-1], Image.NEAREST))

            # Plot
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(norm_in)
            axes[0].set_xlabel("(a) Input image", fontsize=16, labelpad=10)
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            axes[1].imshow(gt)
            axes[1].set_xlabel("(b) Ground truth", fontsize=16, labelpad=10)
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            axes[2].imshow(pred)
            axes[2].set_xlabel("(c) Prediction", fontsize=16, labelpad=10)
            axes[2].set_xticks([])
            axes[2].set_yticks([])

            axes[3].imshow(gt_mask)
            axes[3].set_xlabel(
                "(d) Token pseudo-cluster mask", fontsize=16, labelpad=10
            )
            axes[3].set_xticks([])
            axes[3].set_yticks([])

            axes[4].imshow(mask)
            axes[4].set_xlabel(
                f"(e) Predicted tokens' clusters", fontsize=16, labelpad=10
            )
            axes[4].set_xticks([])
            axes[4].set_yticks([])

            fig.tight_layout()
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.savefig(osp.join(work_dir, f"{image_num}.png"))
            plt.close(fig)

            if image_num >= total_images:
                break

    all_freqs = []
    compression_ratios = []
    k_max = 5

    for i, data in enumerate(tqdm(data_loader)):
        data = model.data_preprocessor(data, True)
        inputs = data["inputs"]
        data_samples = data["data_samples"]

        with torch.no_grad():
            out, logits = model(
                inputs, data_samples, mode="predict", return_logits=True
            )

        logits_pred = torch.argmax(logits[0], dim=-1).cpu().numpy().flatten()
        total = logits_pred.size
        values, counts = np.unique(logits_pred, return_counts=True)

        freq = np.zeros(k_max + 1)
        freq[values] = counts
        freq /= total

        non_zero = total - counts[values == 0][0] if 0 in values else total
        compression_ratio = non_zero / total

        all_freqs.append(freq)
        compression_ratios.append(compression_ratio)

    all_freqs = np.array(all_freqs)
    compression_ratios = np.array(compression_ratios)
    plot_classwise_boxplots(all_freqs, osp.join(work_dir, "frequency_percentiles.png"))


COLORS = [
    [0, 0, 0],
    [247, 107, 104],
    [255, 234, 97],
    [99, 180, 246],
    [51, 202, 127],
    [138, 109, 163],
]

SUIM_COLORS = (
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [0, 255, 255],
    [255, 0, 0],
    [255, 0, 255],
    [255, 255, 0],
    [255, 255, 255],
)


RUMEX_COLORS = ([0, 0, 0], [255, 0, 0], [0, 0, 255])
ADE20K_COLORS = (
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
    [0, 0, 0],
)

if __name__ == "__main__":
    main()
