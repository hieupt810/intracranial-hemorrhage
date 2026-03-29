import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch_3d as smp
import torch
from torch.utils.data import DataLoader

from dataset import BrainMRIDataset
from helpers import get_transforms, setup_logging

setup_logging()


def extract_best_slice(image: np.ndarray, mask: np.ndarray, pred: np.ndarray):
    # Sum over H and W to find the amount of mask per depth slice
    # mask[0] removes the channel dimension
    slice_areas = mask[0].sum(axis=(1, 2))

    if slice_areas.max() > 0:
        best_slice_idx = slice_areas.argmax()
    else:
        best_slice_idx = mask.shape[1] // 2

    return (
        image[0, best_slice_idx, :, :],
        mask[0, best_slice_idx, :, :],
        pred[0, best_slice_idx, :, :],
        best_slice_idx,
    )


def plot_and_save_results(
    data_dir: Union[str, Path],
    model_path: str,
    output_dir: str,
    batch_size: int = 1,
    num_workers: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate test dataset and dataloader
    test_dataset = BrainMRIDataset(
        root_dir=data_dir, split="test", transforms=get_transforms(is_training=False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Load Model
    logging.info(f"Loading model from {model_path} onto {device}...")
    model = smp.UnetPlusPlus(encoder_name="resnet101", in_channels=1, classes=1).to(
        device
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logging.info(f"Starting inference and plotting. Saving to '{output_dir}'...")

    with torch.no_grad():
        for batch_idx, (inputs, masks) in enumerate(test_loader):
            inputs = inputs.to(device)

            # Forward pass
            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = model(inputs)

            # Apply sigmoid to get probabilities, then threshold at 0.5 for binary mask
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Move to CPU and convert to numpy for matplotlib
            inputs_np = inputs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Iterate through the batch to plot each volume
            for i in range(inputs.size(0)):
                global_idx = batch_idx * batch_size + i

                # Extract the most informative 2D slice from the 3D volume
                img_slice, mask_slice, pred_slice, z_idx = extract_best_slice(
                    inputs_np[i], masks_np[i], preds_np[i]
                )

                # Plotting
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(
                    f"Sample {global_idx:03d} | Depth Slice Z={z_idx}", fontsize=14
                )

                # Image
                axes[0].imshow(img_slice, cmap="gray")
                axes[0].set_title("MRI Image")
                axes[0].axis("off")

                # Ground Truth Mask
                axes[1].imshow(img_slice, cmap="gray")
                axes[1].imshow(mask_slice, cmap="Greens", alpha=0.5)  # Overlay
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                # Predicted Mask
                axes[2].imshow(img_slice, cmap="gray")
                axes[2].imshow(pred_slice, cmap="Reds", alpha=0.5)  # Overlay
                axes[2].set_title("Predicted Mask")
                axes[2].axis("off")

                plt.tight_layout()

                # Save the figure
                save_path = os.path.join(output_dir, f"result_{global_idx:03d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

    logging.info(f"Successfully generated plots for {len(test_dataset)} test samples.")


def setup_args():
    parser = ArgumentParser(description="Evaluate and plot 3D U-Net++ predictions.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="Path to trained weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_plots",
        help="Directory to save the plots.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of worker threads."
    )
    return parser.parse_args()


def main():
    args = setup_args()
    plot_and_save_results(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
