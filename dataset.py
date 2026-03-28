from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(
        self, root_dir: Union[str, Path], split: str = "train", transforms=None
    ):
        self.dir = Path(root_dir) / split
        self.patient_dirs = [d for d in self.dir.iterdir() if d.is_dir()]
        self.transforms = transforms

    def __len__(self):
        return len(self.patient_dirs)

    def _load_and_stack(self, dir: Union[str, Path]):
        paths = sorted(Path(dir).glob("*.png"))

        # Read each image as grayscale and convert to numpy array
        slices = [np.array(Image.open(p).convert("L"), dtype=np.float32) for p in paths]

        # Stack into 3D volume (D, H, W)
        volume = np.stack(slices, axis=0)
        return volume

    def __getitem__(self, index):
        patient_dir = self.patient_dirs[index]

        img_dir = Path(patient_dir) / "images"
        mask_dir = Path(patient_dir) / "masks"

        image_volume = self._load_and_stack(img_dir)
        mask_volume = self._load_and_stack(mask_dir)

        # Add channel dimension: Shape becomes (C, D, H, W)
        image_volume = np.expand_dims(image_volume, axis=0)
        mask_volume = np.expand_dims(mask_volume, axis=0)

        # Create dictionary for MONAI transforms
        data = {"image": image_volume, "mask": mask_volume}
        if self.transforms:
            data = self.transforms(data)

        return data
