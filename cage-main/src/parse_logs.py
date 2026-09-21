from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def process_csv_log(csv_path: Path):
    """Parses summary metrics from a CSV log file and saves fold performance plots."""
    if not csv_path.exists():
        print(f"Error: CSV log file not found at {csv_path}")
        return

    output_dir = csv_path.parent / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "test_set" not in df.columns:
        print(
            f"Error: 'test_set' column missing in {csv_path.name}. Found"
            f" columns: {list(df.columns)}"
        )
        return

    print(f"Processing CSV log: {csv_path.name}")

    # Plot AUC performance across dev sets for each test set split
    for test_fold, test_df in df.groupby("test_set"):
        test_dir = output_dir / f"Test_Fold_{test_fold}"
        test_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(8, 5))
        x_labels = test_df["dev_set"].astype(str)

        plt.plot(
            x_labels,
            test_df["dev_auc"],
            marker="o",
            label="Dev AUC",
            color="#ff7f0e",
            linewidth=2,
        )
        plt.plot(
            x_labels,
            test_df["test_auc"],
            marker="s",
            label="Test AUC",
            color="#2ca02c",
            linewidth=2,
        )

        plt.title(f"Fold AUC Performance | Test Fold {test_fold}")
        plt.xlabel("Dev Fold")
        plt.ylabel("AUC")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            test_dir / f"auc_summary_test_fold_{test_fold}.png", dpi=300
        )
        plt.close()

    # Output experiment summary statistics across all folds
    mean_metrics = df[["dev_acc", "dev_auc", "test_acc", "test_auc"]].mean()
    print(f"\n--- Summary Metrics ({csv_path.name}) ---")
    print(f"Mean Dev Acc:  {mean_metrics['dev_acc']:.4f}")
    print(f"Mean Dev AUC:  {mean_metrics['dev_auc']:.4f}")
    print(f"Mean Test Acc: {mean_metrics['test_acc']:.4f}")
    print(f"Mean Test AUC: {mean_metrics['test_auc']:.4f}")
    print(f"Plots saved to directory: '{output_dir}'\n")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir.parent / "logs"

    # Automatically find and process all .csv files in /cage-main/logs/
    csv_files = sorted(list(logs_dir.glob("*.csv")))

    if not csv_files:
        print(f"No CSV files found in directory: {logs_dir}")
    else:
        for csv_file in csv_files:
            process_csv_log(csv_file)
