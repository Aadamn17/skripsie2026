# ------------------------------------------------------------------
# report.py
#
# Honest aggregation of the per-fold CSVs produced by main.py.
#
# main.py writes one row per (test_set, dev_set, epoch). The test_auc
# column is non-NaN on exactly one epoch per (test_set, dev_set) pair.
# So a completed run has 10 test folds x 9 dev folds = 90 non-NaN
# test_auc values.
#
# Those 90 values are NOT independent: the 9 dev-fold runs that share
# a test fold all test on the same patients. Reporting mean +/- std
# over all 90 understates the true std and mis-states what the mean is.
#
# This script:
#   1. collapses the 9 dev-fold runs within each test fold into a
#      single per-test-fold mean;
#   2. reports the mean +/- std over the 10 per-test-fold means;
#   3. also prints a 95% CI on that mean;
#   4. prints the old (naive 90-row) aggregation for comparison.
#
# Usage:
#   python report.py results/your_file.csv [results/other_file.csv ...]
# ------------------------------------------------------------------
import argparse
import sys

import numpy as np
import pandas as pd


def summarise(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[{csv_path}] file not found", file=sys.stderr)
        return None

    required = {"test_set", "dev_set", "test_auc"}
    if not required.issubset(df.columns):
        print(f"[{csv_path}] missing columns {required - set(df.columns)}",
              file=sys.stderr)
        return None

    # Only the rows where a test AUC was actually recorded.
    test_rows = df.dropna(subset=["test_auc"]).copy()
    if test_rows.empty:
        print(f"[{csv_path}] no non-NaN test_auc rows — did training finish?",
              file=sys.stderr)
        return None

    # ---- Step 1: collapse dev folds within each test fold ----
    per_test_fold = (
        test_rows
        .groupby("test_set")["test_auc"]
        .agg(mean_auc="mean", std_auc="std", n_dev_folds="count")
        .sort_index()
    )

    print(f"\n=== {csv_path} ===")
    print(f"non-NaN test_auc rows: {len(test_rows)} "
          f"({per_test_fold['n_dev_folds'].sum()} across "
          f"{len(per_test_fold)} test folds)")
    print("\nPer-test-fold summary (mean over dev folds):")
    print(per_test_fold.round(4).to_string())

    # ---- Step 2: headline = mean +/- std over the 10 test-fold means ----
    test_means = per_test_fold["mean_auc"].values
    n_test     = len(test_means)

    if n_test < 2:
        print(f"\nOnly {n_test} test fold(s) present — cannot compute a std.")
        return float(test_means[0])

    headline_mean = float(test_means.mean())
    headline_std  = float(test_means.std(ddof=1))   # sample std across test folds
    ci95          = 1.96 * headline_std / np.sqrt(n_test)

    print(f"\nHeadline test AUC: {headline_mean:.4f} ± {headline_std:.4f} "
          f"(n = {n_test} test folds)")
    print(f"95% CI on the mean: "
          f"[{headline_mean - ci95:.4f}, {headline_mean + ci95:.4f}]")

    # ---- Old aggregation, for comparison ----
    naive_mean = float(test_rows["test_auc"].mean())
    naive_std  = float(test_rows["test_auc"].std(ddof=1))
    print(f"\n(for comparison, naive {len(test_rows)}-row aggregation: "
          f"{naive_mean:.4f} ± {naive_std:.4f})")

    return headline_mean


def main():
    ap = argparse.ArgumentParser(
        description="Honest AUC aggregation over main.py CSVs.")
    ap.add_argument("csv", nargs="+",
                    help="one or more CSV files produced by main.py")
    args = ap.parse_args()

    for path in args.csv:
        summarise(path)


if __name__ == "__main__":
    main()