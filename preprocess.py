import argparse
import logging
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Union

from PIL import Image

from helpers import seed_everything, setup_logging

setup_logging()


def process_patient_data(
    patient_dir: Path,
    output_dir: Path,
    target_count: int = 15,
    image_size: tuple[int, int] = (512, 512),
):
    if not patient_dir.exists():
        logging.error("Patient directory %s does not exist.", patient_dir)
        return

    # Create output directories
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    used_stems = set()
    saved_count = 0

    # 1. Process images with corresponding masks
    mask_paths = sorted(patient_dir.glob("*_HGE_Seg.jpg"))
    for mask_path in mask_paths:
        if saved_count >= target_count:
            break

        # Derive the corresponding image path by removing the "_HGE_Seg" suffix
        base_stem = mask_path.stem.replace("_HGE_Seg", "")
        image_path = mask_path.with_name(base_stem + mask_path.suffix)

        if not image_path.exists():
            logging.warning("Image file %s missing for mask %s.", image_path, mask_path)
            continue

        try:
            # Use context managers to prevent memory leaks from unclosed files
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                img_resized = image.resize(image_size)
                msk_resized = mask.resize(image_size)

                saved_count += 1
                img_resized.save(images_dir / f"{saved_count}.png")
                msk_resized.save(masks_dir / f"{saved_count}.png")

            used_stems.add(base_stem)
        except Exception as e:
            logging.error("Error processing %s and %s: %s", image_path, mask_path, e)

    # 2. Pad with empty masks if target count is not reached
    if saved_count < target_count:
        empty_mask = Image.new("L", image_size, 0)
        images_path = sorted(patient_dir.glob("*.jpg"))

        for image_path in images_path:
            if saved_count >= target_count:
                break

            # Skip if it is a mask file or an already processed image
            if image_path.stem.endswith("_HGE_Seg") or image_path.stem in used_stems:
                continue

            try:
                with Image.open(image_path) as image:
                    img_resized = image.resize(image_size)

                    saved_count += 1
                    img_resized.save(images_dir / f"{saved_count}.png")
                    empty_mask.save(masks_dir / f"{saved_count}.png")

                used_stems.add(image_path.stem)
            except Exception as e:
                logging.error("Error processing %s: %s", image_path, e)

    if saved_count < target_count:
        logging.warning(
            "Only processed %d/%d images for patient %s.",
            saved_count,
            target_count,
            patient_dir.name,
        )


def process_patients(
    directories: list[Path],
    processed_data_dir: Path,
    target_count: int,
    num_workers: int = 4,
):
    """Processes multiple patients in parallel."""
    # Ensure all target directories are created before parallel execution
    for d in directories:
        (processed_data_dir / d.parent.stem).mkdir(parents=True, exist_ok=True)

    # Use multiprocessing to speed up the I/O and resizing tasks
    process_func = partial(
        process_patient_wrapper,
        processed_data_dir=processed_data_dir,
        target_count=target_count,
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, d) for d in directories]
        for future in as_completed(futures):
            future.result()


def process_patient_wrapper(
    patient_dir: Path, processed_data_dir: Path, target_count: int
):
    """Helper function to unpack arguments for multiprocessing."""
    processed_dir = processed_data_dir / patient_dir.parent.stem
    process_patient_data(patient_dir, processed_dir, target_count=target_count)


def process_dataset(
    raw_data_dir: Union[str, Path],
    processed_data_dir: Union[str, Path],
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_count: int = 15,
    seed: int = 42,
    overwrite: bool = False,
    workers: int = 4,
):
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logging.error("Raw data directory %s does not exist.", raw_data_dir)
        return

    processed_data_dir = Path(processed_data_dir)

    # Safer directory management
    if processed_data_dir.exists():
        if overwrite:
            logging.info("Overwriting existing processed data directory.")
            shutil.rmtree(processed_data_dir)
        else:
            logging.error(
                "Processed directory %s already exists. Use --overwrite to replace it.",
                processed_data_dir,
            )
            return

    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Find patient directories dynamically
    patient_dirs = [
        d / "brain"
        for d in raw_data_dir.iterdir()
        if d.is_dir() and (d / "brain").exists()
    ]

    if not patient_dirs:
        logging.error("No valid patient directories found in %s.", raw_data_dir)
        return

    num_patients = len(patient_dirs)
    logging.info("Found %d patient directories.", num_patients)

    seed_everything(seed)
    random.shuffle(patient_dirs)

    # Calculate splits safely
    validation_size = int(num_patients * validation_ratio)
    test_size = int(num_patients * test_ratio)
    train_size = num_patients - validation_size - test_size

    train_dirs = patient_dirs[:train_size]
    validation_dirs = patient_dirs[train_size : train_size + validation_size]
    test_dirs = patient_dirs[train_size + validation_size :]

    logging.info("Starting processing. Using %d parallel workers...", workers)

    process_patients(train_dirs, processed_data_dir / "train", target_count, workers)
    process_patients(
        validation_dirs, processed_data_dir / "validation", target_count, workers
    )
    process_patients(test_dirs, processed_data_dir / "test", target_count, workers)

    logging.info(
        "Dataset processing complete. Train: %d, Validation: %d, Test: %d",
        len(train_dirs),
        len(validation_dirs),
        len(test_dirs),
    )


def setup_args():
    parser = argparse.ArgumentParser(
        description="Preprocess intracranial hemorrhage dataset."
    )
    parser.add_argument(
        "--raw_data_dir", type=str, required=True, help="Path to raw data."
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="processed_dataset",
        help="Output path.",
    )
    parser.add_argument(
        "--target_count", type=int, default=19, help="Images per patient."
    )
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output dir if it exists."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_args()
    process_dataset(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        target_count=args.target_count,
        seed=args.seed,
        overwrite=args.overwrite,
        workers=args.workers,
    )
