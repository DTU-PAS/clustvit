import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from scipy.ndimage import label


@METRICS.register_module()
class GlobalSizeStratifiedRecall(BaseMetric):
    def __init__(self, num_classes, ignore_index=255, thresholds_pct=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Default relative area thresholds (percentage of total image area)
        if thresholds_pct is None:
            self.thresholds_pct = {
                "small": (0.0, 0.05),
                "medium": (0.05, 0.15),
                "large": (0.15, 0.30),
                "huge": (0.30, float("inf")),
            }
        else:
            self.thresholds_pct = thresholds_pct

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            # 1. Safely extract predictions and GT whether it is a dict or an object
            if isinstance(data_sample, dict):
                pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
                gt_label = data_sample["gt_sem_seg"]["data"].squeeze()
            else:
                pred_label = data_sample.pred_sem_seg.data.squeeze()
                gt_label = data_sample.gt_sem_seg.data.squeeze()

            # 2. Ensure they are numpy arrays
            if hasattr(pred_label, "cpu"):
                pred_label = pred_label.cpu().numpy()
            if hasattr(gt_label, "cpu"):
                gt_label = gt_label.cpu().numpy()

            # Calculate total image area to determine relative percentages
            image_area = gt_label.shape[0] * gt_label.shape[1]

            # Initialize counts for this image ONLY by size bucket (class is ignored here)
            img_results = {s: {"TP": 0, "Total": 0} for s in self.thresholds_pct}

            for class_id in range(self.num_classes):
                if class_id == self.ignore_index:
                    continue

                # Temporarily isolate the class just to get accurate object boundaries
                gt_class_mask = (gt_label == class_id).astype(int)
                labeled_array, num_features = label(gt_class_mask)

                for obj_idx in range(1, num_features + 1):
                    obj_mask = labeled_array == obj_idx
                    area = obj_mask.sum()
                    obj_pct = area / image_area

                    size_bucket = None
                    for bucket, (min_pct, max_pct) in self.thresholds_pct.items():
                        if min_pct <= obj_pct < max_pct:
                            size_bucket = bucket
                            break

                    if size_bucket is None:
                        continue

                    # Count the pixels
                    total_pixels = area
                    tp_pixels = (obj_mask & (pred_label == class_id)).sum()

                    # Dump the results directly into the global size bucket
                    img_results[size_bucket]["Total"] += total_pixels
                    img_results[size_bucket]["TP"] += tp_pixels

            self.results.append(img_results)

    def compute_metrics(self, results: list) -> dict:
        # Aggregate totals across the entire dataset globally
        aggregated = {s: {"TP": 0, "Total": 0} for s in self.thresholds_pct}

        for img_res in results:
            for size_bucket, counts in img_res.items():
                aggregated[size_bucket]["TP"] += counts["TP"]
                aggregated[size_bucket]["Total"] += counts["Total"]

        metrics = {}

        # Compute the final global recall per bucket
        for size_bucket in self.thresholds_pct:
            tp = aggregated[size_bucket]["TP"]
            total = aggregated[size_bucket]["Total"]

            metric_name = f"Global_Recall_{size_bucket}"
            if total > 0:
                metrics[metric_name] = np.round((tp / total) * 100, 2)
            else:
                metrics[metric_name] = float("nan")

        return metrics
