import csv
import logging
import shutil
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
from monai.losses import DiceFocalLoss
from monai.networks.nets import UNet
from sklearn.model_selection import KFold
from torch import amp, optim
from torch.utils.data import DataLoader, Subset

from dataset import BrainMRIDataset
from helpers import get_transforms, seed_everything, setup_logging
from metrics import compute_metrics

warnings.filterwarnings("ignore")
setup_logging()


def _create_model(device: torch.device) -> UNet:
    """Create a fresh UNet model."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


def _train_one_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[float, dict]:
    """Train a single fold and return (best_val_loss, validation_metrics)."""

    model = _create_model(device)
    criterion = DiceFocalLoss(include_background=False, softmax=True, to_onehot_y=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = amp.grad_scaler.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    model_save_path = f"best_model_fold{fold}.pth"

    for epoch in range(epochs):
        losses = {"train": 0.0, "val": 0.0}

        for phase in ["train", "val"]:
            is_train = phase == "train"
            loader = train_loader if is_train else val_loader
            model.train(is_train)

            for batch_data in loader:
                if isinstance(batch_data, list):
                    inputs = torch.cat([b["image"] for b in batch_data], dim=0).to(
                        device
                    )
                    masks = torch.cat([b["mask"] for b in batch_data], dim=0).to(device)
                else:
                    inputs, masks = (
                        batch_data["image"].to(device),
                        batch_data["mask"].to(device),
                    )

                if is_train:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    with torch.autocast(
                        device_type=device.type, enabled=(device.type == "cuda")
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, masks)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                losses[phase] += loss.item() * inputs.size(0)

            losses[phase] /= len(loader.dataset)

        logging.info(
            f"Fold {fold} | Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), model_save_path)

    # --- Evaluate best model on validation set ---
    model.load_state_dict(
        torch.load(model_save_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "hausdorff95": 0.0,
    }
    n_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(batch_data, list):
                inputs = torch.cat([b["image"] for b in batch_data], dim=0).to(device)
                masks = torch.cat([b["mask"] for b in batch_data], dim=0).to(device)
            else:
                inputs, masks = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                outputs = model(inputs)

            # Convert predictions to one-hot
            preds_argmax = torch.argmax(outputs, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(outputs)
            preds_onehot.scatter_(1, preds_argmax, 1)

            # Convert targets to one-hot
            targets_onehot = torch.zeros_like(outputs)
            targets_onehot.scatter_(1, masks.long(), 1)

            batch_metrics = compute_metrics(preds_onehot, targets_onehot)
            for k in all_metrics:
                all_metrics[k] += batch_metrics[k]
            n_batches += 1

    # Average over batches
    for k in all_metrics:
        all_metrics[k] /= max(n_batches, 1)

    logging.info(
        f"Fold {fold} Validation Metrics: "
        f"Dice={all_metrics['dice']:.4f} | IoU={all_metrics['iou']:.4f} | "
        f"Precision={all_metrics['precision']:.4f} | Recall={all_metrics['recall']:.4f} | "
        f"Hausdorff95={all_metrics['hausdorff95']:.4f}"
    )

    return best_val_loss, all_metrics


def kfold_train(
    data_dir: Union[str, Path],
    n_folds: int = 5,
    batch_size: int = 4,
    epochs: int = 100,
    lr: float = 0.001,
    num_workers: int = 1,
    seed: int = 42,
):
    """Run K-Fold cross-validation and select the best model.

    Args:
        data_dir: Path to the processed dataset containing an 'all/' subdirectory.
        n_folds: Number of folds for cross-validation.
        batch_size: Batch size for DataLoaders.
        epochs: Number of training epochs per fold.
        lr: Learning rate.
        num_workers: Number of DataLoader workers.
        seed: Random seed for reproducibility.

    Returns:
        Path to the best model weights file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    # Load the full dataset (all patients in one directory)
    full_dataset = BrainMRIDataset(root_dir=data_dir, transforms=None)

    n_patients = len(full_dataset)
    logging.info(f"Starting {n_folds}-Fold CV on {n_patients} patients on {device}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    best_fold = -1
    best_dice = -1.0

    for fold, (train_indices, val_indices) in enumerate(kf.split(range(n_patients)), 1):
        logging.info(f"{'=' * 60}")
        logging.info(
            f"FOLD {fold}/{n_folds} — Train: {len(train_indices)}, Val: {len(val_indices)}"
        )
        logging.info(f"{'=' * 60}")

        # Create per-fold datasets with appropriate transforms
        train_dataset = BrainMRIDataset(
            root_dir=data_dir, transforms=get_transforms(is_training=True)
        )
        val_dataset = BrainMRIDataset(
            root_dir=data_dir, transforms=get_transforms(is_training=False)
        )

        train_subset = Subset(train_dataset, train_indices.tolist())
        val_subset = Subset(val_dataset, val_indices.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        val_loss, metrics = _train_one_fold(
            fold=fold,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        fold_results.append(
            {
                "fold": fold,
                "val_loss": val_loss,
                **metrics,
            }
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_fold = fold

    # --- Summary ---
    logging.info(f"\n{'=' * 60}")
    logging.info("K-FOLD CROSS-VALIDATION RESULTS")
    logging.info(f"{'=' * 60}")

    metric_keys = ["dice", "iou", "precision", "recall", "hausdorff95"]

    for result in fold_results:
        logging.info(
            f"Fold {result['fold']}: "
            + " | ".join(f"{k}={result[k]:.4f}" for k in metric_keys)
        )

    # Compute mean and std
    means = {k: np.mean([r[k] for r in fold_results]) for k in metric_keys}
    stds = {k: np.std([r[k] for r in fold_results]) for k in metric_keys}

    logging.info(
        "Mean:  " + " | ".join(f"{k}={means[k]:.4f}±{stds[k]:.4f}" for k in metric_keys)
    )

    logging.info(f"\nBest fold: {best_fold} (Dice={best_dice:.4f})")

    # Copy best fold model to best_model.pth
    best_fold_path = f"best_model_fold{best_fold}.pth"
    shutil.copy2(best_fold_path, "best_model.pth")
    logging.info(f"Copied {best_fold_path} → best_model.pth")

    # Save results to CSV
    csv_path = "kfold_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "val_loss"] + metric_keys)
        writer.writeheader()
        writer.writerows(fold_results)

        # Write mean/std row
        mean_row = {
            "fold": "mean",
            "val_loss": np.mean([r["val_loss"] for r in fold_results]),
        }
        mean_row.update(means)
        writer.writerow(mean_row)

        std_row = {
            "fold": "std",
            "val_loss": np.std([r["val_loss"] for r in fold_results]),
        }
        std_row.update(stds)
        writer.writerow(std_row)

    logging.info(f"Results saved to {csv_path}")

    return "best_model.pth"
