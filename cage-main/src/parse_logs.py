from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def process_csv_log(csv_path: Path):
    """Parses metrics from a CSV log file and saves per-combo and mean plots."""
    if not csv_path.exists():
        print(f"Error: CSV log file not found at {csv_path}")
        return

    output_dir = csv_path.parent / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {
        "test_set",
        "dev_set",
        "epoch",
        "train_loss",
        "dev_loss",
        "test_loss",
        "dev_acc",
        "dev_auc",
        "test_acc",
        "test_auc",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"Error: Missing columns in {csv_path.name}: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
        return

    print(f"Processing CSV log: {csv_path.name}")

    # Collapse duplicated runs of the same (test_set, dev_set, epoch) by averaging.
    df = (
        df.groupby(["test_set", "dev_set", "epoch"], as_index=False)
        .agg(
            train_loss=("train_loss", "mean"),
            dev_loss=("dev_loss", "mean"),
            test_loss=("test_loss", "mean"),
            dev_acc=("dev_acc", "mean"),
            dev_auc=("dev_auc", "mean"),
            test_acc=("test_acc", "mean"),
            test_auc=("test_auc", "mean"),
        )
    )

    # ----- Per (test_set, dev_set) plots -----
    for test_fold, test_df in df.groupby("test_set"):
        test_dir = output_dir / f"Test_Fold_{test_fold}"
        test_dir.mkdir(exist_ok=True)

        for dev_fold, dev_df in test_df.groupby("dev_set"):
            dev_df = dev_df.sort_values("epoch")

            # ---- Loss curves ----
            plt.figure(figsize=(8, 5))
            plt.plot(
                dev_df["epoch"],
                dev_df["train_loss"],
                marker="o",
                linewidth=2,
                label="Train Loss",
            )
            plt.plot(
                dev_df["epoch"],
                dev_df["dev_loss"],
                marker="s",
                linewidth=2,
                label="Dev Loss",
            )
            plt.plot(
                dev_df["epoch"],
                dev_df["test_loss"],
                marker="^",
                linewidth=2,
                label="Test Loss",
            )
            plt.title(f"Loss Curves | Test Fold {test_fold} - Dev Fold {dev_fold}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                test_dir / f"loss_test{test_fold}_dev{dev_fold}.png", dpi=300
            )
            plt.close()

            # ---- AUC curves ----
            plt.figure(figsize=(8, 5))
            plt.plot(
                dev_df["epoch"],
                dev_df["dev_auc"],
                marker="o",
                linewidth=2,
                label="Dev AUC",
                color="#ff7f0e",
            )
            plt.plot(
                dev_df["epoch"],
                dev_df["test_auc"],
                marker="s",
                linewidth=2,
                label="Test AUC",
                color="#2ca02c",
            )
            plt.title(f"AUC Curves | Test Fold {test_fold} - Dev Fold {dev_fold}")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                test_dir / f"auc_test{test_fold}_dev{dev_fold}.png", dpi=300
            )
            plt.close()

        # ---- Mean AUC across dev folds for this test fold ----
        mean_df = (
            test_df.groupby("epoch", as_index=False)
            .agg(dev_auc=("dev_auc", "mean"), test_auc=("test_auc", "mean"))
            .sort_values("epoch")
        )

        plt.figure(figsize=(8, 5))
        plt.plot(
            mean_df["epoch"],
            mean_df["dev_auc"],
            marker="o",
            linewidth=2,
            label="Mean Dev AUC",
            color="#ff7f0e",
        )
        plt.plot(
            mean_df["epoch"],
            mean_df["test_auc"],
            marker="s",
            linewidth=2,
            label="Mean Test AUC",
            color="#2ca02c",
        )
        plt.title(f"Mean AUC Across Dev Folds | Test Fold {test_fold}")
        plt.xlabel("Epoch")
        plt.ylabel("Mean AUC")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            test_dir / f"mean_auc_test_fold_{test_fold}.png", dpi=300
        )
        plt.close()

    # ----- Overall summary across the entire CSV -----
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

    csv_files = sorted(logs_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in directory: {logs_dir}")
    else:
        for csv_file in csv_files:
            process_csv_log(csv_file)