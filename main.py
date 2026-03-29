from evaluate import plot_and_save_results
from helpers import setup_args
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
    plot_and_save_results(
        data_dir=args.processed_data_dir,
        model_path="best_model.pth",
        output_dir="./test_plots",
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
