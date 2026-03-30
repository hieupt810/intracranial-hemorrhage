import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from torch.utils.data import DataLoader


def plot_and_save_results(
    dataset,
    model_path,
    output_dir,
    batch_size=1,
    num_workers=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset and DataLoader
    if len(dataset) == 0:
        logging.warning("Test dataset is empty. Cannot evaluate.")
        return

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Initialize model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    if Path(model_path).exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        logging.info(f"Loaded model weights from {model_path}")
    else:
        logging.warning(
            f"Model weights not found at {model_path}. Trying uninitialized model."
        )

    model.eval()

    # We use sliding window inferer because the test volumes might not have spatial dimensions
    # perfectly divisible by 16 (which UNet requires based on the strides).
    inferer = SlidingWindowInferer(
        roi_size=(16, 128, 128), sw_batch_size=4, overlap=0.25
    )

    logging.info("Starting evaluation on test set...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            for idx, batch_data in enumerate(test_loader):
                inputs, masks = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )

                outputs = inferer(inputs, model)
                preds = torch.argmax(outputs, dim=1, keepdim=True)

                # Move back to CPU for plotting
                inputs = inputs.cpu().numpy()
                masks = masks.cpu().numpy()
                preds = preds.cpu().numpy()

                for i in range(inputs.shape[0]):
                    img = inputs[i, 0]  # shape: (D, H, W)
                    gt = masks[i, 0]  # shape: (D, H, W)
                    pr = preds[i, 0]  # shape: (D, H, W)

                    # Get the middle slice along the depth dimension (D)
                    mid_idx = img.shape[0] // 2

                    img_slice = img[mid_idx]
                    gt_slice = gt[mid_idx]
                    pr_slice = pr[mid_idx]

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # 1. Original Image
                    axes[0].imshow(img_slice, cmap="gray")
                    axes[0].set_title("Original MRI Slice")
                    axes[0].axis("off")

                    # 2. Ground Truth Overlay
                    axes[1].imshow(img_slice, cmap="gray")
                    axes[1].imshow(gt_slice, cmap="inferno", alpha=0.5)
                    axes[1].set_title("Ground Truth Mask Overlay")
                    axes[1].axis("off")

                    # 3. Predicted Mask Overlay
                    axes[2].imshow(img_slice, cmap="gray")
                    axes[2].imshow(pr_slice, cmap="inferno", alpha=0.5)
                    axes[2].set_title("Predicted Mask Overlay")
                    axes[2].axis("off")

                    plt.tight_layout()

                    # Save plot
                    save_path = output_path / f"batch{idx}_idx{i}.png"
                    plt.savefig(save_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)

    logging.info(f"Evaluation complete. Extracted plots saved in {output_path}")
