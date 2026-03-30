from dataset import BrainMRIDataset
from evaluate import evaluate_model_metrics, plot_and_save_results
from helpers import get_transforms, setup_args
from kfold_train import kfold_train
from preprocess import process_kfold_dataset


def main():
    args = setup_args()

    # Step 1: Preprocess all patients into a single directory for K-Fold
    process_kfold_dataset(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        target_count=args.target_count,
        seed=args.seed,
        overwrite=args.overwrite,
        workers=args.workers,
    )

    # Step 2: K-Fold cross-validation — trains K models, selects the best
    best_model_path = kfold_train(
        data_dir=args.processed_data_dir,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.workers,
        seed=args.seed,
    )

    # Step 3: Evaluate best model on the full dataset with sliding window
    full_dataset = BrainMRIDataset(
        root_dir=args.processed_data_dir,
        split="all",
        transforms=get_transforms(is_training=False),
    )
    metrics = evaluate_model_metrics(
        dataset=full_dataset,
        model_path=best_model_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print("\n=== Best Model Performance (Full Dataset) ===")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Step 4: Generate visualization plots
    plot_and_save_results(
        dataset=full_dataset,
        model_path=best_model_path,
        output_dir="./evaluation_plots",
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
