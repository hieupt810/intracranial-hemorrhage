import logging
import warnings
from pathlib import Path
from typing import Union

import torch
from monai.losses import DiceFocalLoss
from monai.networks.nets import UNet
from torch import amp, optim
from torch.utils.data import DataLoader

from dataset import BrainMRIDataset
from helpers import get_transforms, seed_everything, setup_logging

warnings.filterwarnings("ignore")
setup_logging()


def train(
    data_dir: Union[str, Path],
    batch_size: int,
    epochs: int,
    lr: float = 0.001,
    num_workers: int = 1,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed for reproducibility
    seed_everything(seed)

    # Instantiate datasets
    train_dataset = BrainMRIDataset(
        root_dir=data_dir, split="train", transforms=get_transforms(is_training=True)
    )
    val_dataset = BrainMRIDataset(
        root_dir=data_dir,
        split="validation",
        transforms=get_transforms(is_training=False),
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    criterion = DiceFocalLoss(include_background=False, softmax=True, to_onehot_y=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = amp.grad_scaler.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    logging.info(f"Starting training on device: {device}")
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
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), "best_model.pth")
            logging.info("Saved best model.")

    logging.info("Training complete.")
