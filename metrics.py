import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute segmentation metrics for a batch of predictions and targets.

    Args:
        preds: Predicted segmentation masks, shape (B, C, D, H, W) — one-hot or softmax.
        targets: Ground truth masks, shape (B, C, D, H, W) — one-hot encoded.

    Returns:
        Dictionary with mean Dice, IoU, Precision, Recall, and Hausdorff95.
    """
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch")
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="mean_batch"
    )

    # Compute Dice and IoU
    dice = dice_metric(preds, targets)
    iou = iou_metric(preds, targets)

    # Compute Hausdorff95 — may fail if prediction or target is empty
    try:
        hausdorff = hausdorff_metric(preds, targets)
        hausdorff_val = hausdorff.nanmean().item()
    except Exception:
        hausdorff_val = float("nan")

    # Compute Precision and Recall manually (exclude background channel 0)
    pred_fg = preds[:, 1:].float()
    target_fg = targets[:, 1:].float()

    tp = (pred_fg * target_fg).sum()
    fp = (pred_fg * (1 - target_fg)).sum()
    fn = ((1 - pred_fg) * target_fg).sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()

    return {
        "dice": dice.nanmean().item(),
        "iou": iou.nanmean().item(),
        "precision": precision,
        "recall": recall,
        "hausdorff95": hausdorff_val,
    }
