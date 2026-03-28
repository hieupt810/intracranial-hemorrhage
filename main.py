import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import segmentation_models_pytorch_3d as smp
import torch
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ToTensord,
)
from torch import amp, optim
from torch.utils.data import DataLoader

from criterion import FocalDiceLoss
from dataset import BrainMRIDataset
from helpers import setup_logging

setup_logging()


def get_transforms(is_training: bool = True):
    transforms = []
    if is_training:
        transforms.extend(
            [
                # Randomly flip along the depth, height, or width axes
                RandFlipd(keys=["image", "mask"], spatial_axis=[0, 1, 2], prob=0.5),
                # Random slight rotations in 3D space
                RandRotated(
                    keys=["image", "mask"],
                    range_x=0.2,
                    range_y=0.2,
                    range_z=0.2,
                    prob=0.4,
                    mode=("bilinear", "nearest"),
                ),
                # Random zoom (scaling)
                RandZoomd(
                    keys=["image", "mask"],
                    prob=0.3,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    mode=("bilinear", "nearest"),
                ),
            ]
        )

    transforms.extend(
        [
            NormalizeIntensityd(keys=["image"], nonzero=True),
            ToTensord(keys=["image", "mask"]),
        ]
    )
    return Compose(transforms)


def train(
    data_dir: Union[str, Path],
    batch_size: int,
    epochs: int,
    lr: float = 1e-4,
    patience: int = 5,
    num_workers: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate datasets
    train_dataset = BrainMRIDataset(
        root_dir=data_dir, split="train", transforms=get_transforms(is_training=True)
    )
    val_dataset = BrainMRIDataset(
        root_dir=data_dir, split="val", transforms=get_transforms(is_training=False)
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model, loss, optimizer, scheduler
    model = smp.UnetPlusPlus(encoder_name="resnet101", in_channels=1, classes=1).to(
        device
    )
    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    # Scaler for mixed precision training
    scaler = amp.grad_scaler.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0

    logging.info(f"Starting training on device: {device}")
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1:03d}/{epochs:03d}")

        # Train phase
        model.train()
        train_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()

            # Forward pass with Mixed Precision
            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                outputs = model(inputs)
                loss = criterion(outputs, masks)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                with torch.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        logging.info(f"\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Step scheduler
        scheduler.step(val_loss)

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, "best_model.pth")
        else:
            patience_counter += 1
            logging.info(f"\tNo improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logging.warning(f"\tEarly stopping triggered after {epoch + 1} epochs.")
                break

    logging.info("Training complete.")


def setup_args():
    parser = ArgumentParser(
        description="Train a 3D U-Net++ model for brain MRI segmentation."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for learning rate scheduler and early stopping.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker threads.",
    )


def main():
    args = setup_args()
    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
    )
