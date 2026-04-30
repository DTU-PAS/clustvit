# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmseg.registry import MODELS

import dynseg

# import loss


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg benchmark a model")
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
        mkdir_or_exist(osp.abspath(args.work_dir))
        npz_file = osp.join(args.work_dir, f"num_tokens_clustvit.npz")
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
        mkdir_or_exist(osp.abspath(work_dir))
        npz_file = osp.join(work_dir, f"num_tokens_clustvit.npz")

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None

    if "injection_points" in cfg.model.backbone:
        injection_point = cfg.model.backbone.injection_points[0]
    else:
        injection_point = 3

    # build the dataloader
    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)

    if "checkpoint" in args and osp.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    if torch.cuda.is_available():
        model = model.cuda()

    model = revert_sync_batchnorm(model)

    model.eval()
    if cfg.test_dataloader.dataset.type == "ADE20KDataset":
        total_iters = 2000
    elif cfg.test_dataloader.dataset.type == "SuimDataset":
        total_iters = 110
    elif cfg.test_dataloader.dataset.type == "RumexWeedsDataset":
        total_iters = 1303
    else:
        total_iters = 500

    # benchmark with 200 batches and take the average
    model.eval()
    model.test_cfg.mode = "whole"
    tokens_clustvit = []
    tokens_vit = []
    tokens_cts = []
    unique_dataset_classes = []

    for i, data in enumerate(data_loader):
        data = model.data_preprocessor(data, True)
        inputs = data["inputs"]
        data_samples = data["data_samples"]
        # Fix: ensure inputs is a tensor, not a list of tensors
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
        inputs = inputs.float()
        gt = data_samples[0].gt_sem_seg.data[0].cpu().numpy()
        unique_classes = len(np.unique(gt)[np.unique(gt) != 255])
        with torch.no_grad():
            output, logits = model(
                inputs, data_samples, mode="predict", return_logits=True
            )
            # print("output", output)
        logits_pred = torch.argmax(logits[0], dim=-1).cpu().numpy()
        num_of_compressed_tokens = (
            len(logits_pred[logits_pred == 0]) + len(np.unique(logits_pred)) - 1
        )
        number_of_tokens_clustvit = len(
            logits_pred[0].flatten()
        ) * injection_point + num_of_compressed_tokens * (12 - injection_point)
        number_of_tokens_vit = len(logits_pred[0].flatten()) * 12
        number_of_tokens_cts = 712 * 12

        tokens_clustvit.append(number_of_tokens_clustvit)
        tokens_vit.append(number_of_tokens_vit)
        tokens_cts.append(number_of_tokens_cts)
        unique_dataset_classes.append(unique_classes)
        if (i + 1) == total_iters:
            break
            # benchmark with 200 batches and take the average

    avg_tokens_clustvit = sum(tokens_clustvit) / len(tokens_clustvit)
    avg_tokens_vit = sum(tokens_vit) / len(tokens_vit)
    avg_tokens_cts = sum(tokens_cts) / len(tokens_cts)

    std_tokens_clustvit = np.std(tokens_clustvit)
    std_tokens_vit = np.std(tokens_vit)
    std_tokens_cts = np.std(tokens_cts)

    avg_tokens_clustvit = np.array(avg_tokens_clustvit)
    np.savez(
        npz_file.replace("num_tokens_clustvit", "dataset_classes_per_image "),
        unique_dataset_classes=unique_dataset_classes,
    )
    np.savez(
        npz_file,
        tokens_clustvit=tokens_clustvit,
    )

    print(
        f"Average number of tokens: {avg_tokens_clustvit} +- {std_tokens_clustvit}, \t Average number of tokens vit: {avg_tokens_vit} +- {std_tokens_vit}, \t Average number of tokens cts: {avg_tokens_cts} +- {std_tokens_cts}"
    )


if __name__ == "__main__":
    main()
