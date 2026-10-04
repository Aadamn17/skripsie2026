"""Aggregate results CSVs and per-patient predictions."""
import os
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")


def load_results(results_dir=RESULTS_DIR):
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "test_sens" not in df.columns or "test_spec" not in df.columns:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_config_summary(df):
    best = df.dropna(subset=["test_auc"]).copy()
    if best.empty:
        return pd.DataFrame()

    group_cols = ["fusion", "arch", "lr", "wd", "aug", "use_pretrained"]
    grouped = best.groupby(group_cols).agg(
        AUC_mean=("test_auc",  "mean"),  AUC_std=("test_auc",  "std"),
        Acc_mean=("test_acc",  "mean"),  Acc_std=("test_acc",  "std"),
        Sens_mean=("test_sens","mean"),  Sens_std=("test_sens","std"),
        Spec_mean=("test_spec","mean"),  Spec_std=("test_spec","std"),
        Folds=("test_auc", "count"),
    ).round(4).reset_index()
    return grouped.sort_values("AUC_mean", ascending=False)


def threshold_sweep(predictions_dir=PREDICTIONS_DIR,
                    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    files = glob.glob(os.path.join(predictions_dir, "*.csv"))
    if not files:
        return pd.DataFrame()

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if "set" not in df.columns or df.empty:
            continue
        test = df[df["set"] == "test"]
        if test.empty:
            continue
        per_patient = test.groupby("patient_id").agg(
            true_label=("true_label", "first"),
            prob_pos=("prob_pos", "mean"),
        ).reset_index()

        base = {
            "fusion":     df["fusion"].iloc[0],
            "arch":       df["arch"].iloc[0],
            "lr":         df["lr"].iloc[0],
            "aug":        df["aug"].iloc[0],
            "pretrained": df["pretrained"].iloc[0],
        }
        y_true = per_patient["true_label"].values
        y_prob = per_patient["prob_pos"].values

        for t in thresholds:
            y_pred = (y_prob > t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            rows.append({**base, "threshold": t,
                         "sens": round(tp / max(tp + fn, 1), 4),
                         "spec": round(tn / max(tn + fp, 1), 4)})
    return pd.DataFrame(rows)


def main():
    df = load_results()
    if df.empty:
        print(f"No usable results CSVs in {RESULTS_DIR}")
        return

    summary = per_config_summary(df)
    if not summary.empty:
        print("\n" + "=" * 100)
        print("TEST-SET SUMMARY  (mean ± std across completed folds)")
        print("=" * 100)
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
        print(f"\nWrote {RESULTS_DIR}/results_summary.csv")

    sweep = threshold_sweep()
    if not sweep.empty:
        sweep.to_csv(os.path.join(RESULTS_DIR, "threshold_sweep.csv"), index=False)
        print(f"Wrote {RESULTS_DIR}/threshold_sweep.csv")

        if not summary.empty:
            best = summary.iloc[0]
            mask = (
                (sweep["fusion"]     == best["fusion"]) &
                (sweep["arch"]       == best["arch"]) &
                (sweep["lr"]         == best["lr"]) &
                (sweep["aug"]        == best["aug"]) &
                (sweep["pretrained"] == best["use_pretrained"])
            )
            print("\nBest config threshold sweep:")
            print(sweep[mask].to_string(index=False))


if __name__ == "__main__":
    main()