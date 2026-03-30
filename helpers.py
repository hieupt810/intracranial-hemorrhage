def setup_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--processed_data_dir", type=str, default="processed_dataset")
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--target_count", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", type=bool, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)

    return parser.parse_args()


def setup_logging():
    """Set up logging configuration."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int):
    """Set random seed for reproducibility."""
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(is_training: bool = True):
    from monai.transforms import (
        Compose,
        DivisiblePadd,
        EnsureChannelFirstd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotate90d,
        ScaleIntensityRangePercentilesd,
        ToTensord,
    )

    transforms = [
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
        ),
        DivisiblePadd(keys=["image", "mask"], k=16),
    ]

    if is_training:
        transforms.extend(
            [
                RandCropByPosNegLabeld(
                    keys=["image", "mask"],
                    label_key="mask",
                    spatial_size=(16, 128, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "mask"], spatial_axis=[1, 2], prob=0.5),
                RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=[1, 2]),
            ]
        )

    transforms.append(ToTensord(keys=["image", "mask"]))
    return Compose(transforms)
