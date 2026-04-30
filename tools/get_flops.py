# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager

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
from torch.profiler import ProfilerActivity, profile

import dynseg

# import loss

sys.setrecursionlimit(3000)  # Adjust the value as needed
# Suppress DeepSpeed logs
logging.getLogger("DeepSpeed").setLevel(logging.WARNING)


def human_format(num, precision=2):
    """
    Converts a large number into a human-readable format with suffixes.

    Parameters:
    - num (float): The number to format.
    - precision (int): Number of decimal places to include.

    Returns:
    - str: The formatted string.
    """
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{unit}"
        num /= 1000.0
    return f"{num:.{precision}f}E"


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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
        json_file = osp.join(args.work_dir, f"gflops_{timestamp}.json")
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
        mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f"gflops_{timestamp}.json")

    repeat_times = args.repeat_times
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None

    # benchmark_dict = dict(config=args.config, unit="img / s")
    # overall_fps_list = []
    # cfg.test_dataloader.batch_size = 1
    for time_index in range(repeat_times):
        print(f"Run {time_index + 1}:")
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

        gflops_collect = []
        # benchmark with all batches and take the average
        for i, data in enumerate(data_loader):
            data = model.data_preprocessor(data, True)
            inputs = data["inputs"]
            data_samples = data["data_samples"]
            # Fix: ensure inputs is a tensor, not a list of tensors
            if isinstance(inputs, list):
                inputs = torch.stack(inputs, dim=0)
            inputs = inputs.float()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=False,
                with_flops=True,
            ) as prof:
                with torch.no_grad():
                    _ = model(inputs, data_samples, mode="predict")

            total_flops = sum(
                [e.flops for e in prof.key_averages() if e.flops is not None]
            )
            gflops = total_flops / 1e9
            gflops_collect.append(gflops)

            if (i + 1) % args.log_interval == 0:
                avg_gflops = sum(gflops_collect) / len(gflops_collect)
                print(
                    f"Iteration {i + 1}: Current GFLOPs = {gflops:.2f}, Avg GFLOPs = {avg_gflops:.2f}"
                )

        # Final average
        avg_gflops = sum(gflops_collect) / len(gflops_collect)
        std_gflops = np.std(gflops_collect)
        print(f"Final Average GFLOPs over {args.repeat_times} runs: {avg_gflops:.2f}")
        print(f"Final Standard Deviation of GFLOPs: {std_gflops:.2f}")

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(
            np.arange(len(gflops_collect)),
            gflops_collect,
            marker="o",
            linestyle="-",
            color="blue",
        )
        plt.xlabel("Iteration")
        plt.ylabel("GFLOPs")
        plt.title("GFLOPs over Iterations (Torch Profiler)")
        plt.grid(True)
        plt.savefig(f"{work_dir}/gflops_torch_profiler.png")
        plt.close()
        dump(
            {
                "gflops": gflops_collect,
                "avg_gflops": avg_gflops,
                "std_gflops": std_gflops,
            },
            json_file,
            indent=4,
        )


if __name__ == "__main__":
    main()
