from dataset import BrainMRIDataset
from evaluate import plot_and_save_results
from helpers import get_transforms, setup_args
from preprocess import process_dataset
from train import train


def main():
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
    train(
        data_dir=args.processed_data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.workers,
        seed=args.seed,
    )
    validation_dataset = BrainMRIDataset(
        root_dir=args.processed_data_dir,
        split="validation",
        transforms=get_transforms(is_training=False),
    )
    plot_and_save_results(
        dataset=validation_dataset,
        model_path="best_model.pth",
        output_dir="./validation_plots",
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    test_dataset = BrainMRIDataset(
        root_dir=args.processed_data_dir,
        split="test",
        transforms=get_transforms(is_training=False),
    )
    plot_and_save_results(
        dataset=test_dataset,
        model_path="best_model.pth",
        output_dir="./test_plots",
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
