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
        NormalizeIntensityd,
        RandFlipd,
        RandRotated,
        RandZoomd,
        ToTensord,
    )

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
            DivisiblePadd(keys=["image", "mask"], k=32),
            ToTensord(keys=["image", "mask"]),
        ]
    )
    return Compose(transforms)
