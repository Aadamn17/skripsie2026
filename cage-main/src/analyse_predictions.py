"""
Analyse per-patient predictions from a completed grid run.

Reads every CSV in `src/predictions/` and produces:

    analysis/per_fold_pair.csv        one row per (config, test_set, dev_set)
    analysis/per_config_summary.csv   mean ± std per config across fold pairs
    analysis/threshold_sweep.csv      sens/spec at multiple thresholds
    analysis/per_patient.csv          per-patient aggregated predictions
    analysis/error_analysis.csv       systematically misclassified patients
    analysis/significance_tests.csv   pairwise Wilcoxon vs the cough baseline
    analysis/calibration.csv          reliability bins per config

Usage:
    python3 src/analyse_predictions.py
    python3 src/analyse_predictions.py --predictions-dir path/to/preds
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "analysis")

CONFIG_COLS = ["fusion", "arch", "lr", "wd", "aug", "pretrained"]
FOLD_COLS   = ["test_set", "dev_set"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(predictions_dir):
    """Concatenate every prediction CSV into one long DataFrame."""
    files = sorted(glob.glob(os.path.join(predictions_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs found in {predictions_dir}")

    print(f"Found {len(files)} prediction files")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            print(f"  skip {os.path.basename(f)}: {exc}")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique()} unique patients")
    return df


def filter_test(df):
    """Keep only test-set predictions and the columns we need."""
    t = df[df["set"] == "test"].copy()
    # prob_pos is already the model's softmax positive-class probability
    keep = ["patient_id", "true_label", "prob_pos",
            "logit_neg", "logit_pos", "n_coughs"] + CONFIG_COLS + FOLD_COLS
    return t[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return metrics.roc_auc_score(y_true, y_score)


def safe_roc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr), np.trapz(tpr, fpr)


def sens_spec_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def youden_threshold(y_true, y_prob):
    """Threshold that maximises sens + spec - 1 on a dev set."""
    fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def threshold_at_spec(y_true, y_prob, target_spec):
    """Lowest threshold achieving at least `target_spec` specificity."""
    fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
    spec = 1 - fpr
    ok = np.where(spec >= target_spec)[0]
    if len(ok) == 0:
        return float(thr[-1])
    return float(thr[ok[0]])


# ---------------------------------------------------------------------------
# Per-fold-pair metrics
# ---------------------------------------------------------------------------

def per_fold_pair_metrics(df):
    """One row per (config, test_set, dev_set)."""
    rows = []
    groups = df.groupby(CONFIG_COLS + FOLD_COLS, dropna=False)
    for key, g in groups:
        cfg = dict(zip(CONFIG_COLS + FOLD_COLS, key))
        y_true = g["true_label"].values
        y_prob = g["prob_pos"].values
        auc = safe_auc(y_true, y_prob)
        sens, spec = sens_spec_at_threshold(y_true, y_prob, 0.5)
        acc = ((y_prob >= 0.5).astype(int) == y_true).mean()
        brier = np.mean((y_prob - y_true) ** 2)
        rows.append({**cfg,
                     "n_patients": len(g),
                     "n_pos": int(y_true.sum()),
                     "auc": auc,
                     "acc": acc,
                     "sens": sens,
                     "spec": spec,
                     "brier": brier})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-config summary (aggregate over the 90 fold pairs)
# ---------------------------------------------------------------------------

def per_config_summary(per_fold):
    def agg(g):
        return pd.Series({
            "folds":        len(g),
            "auc_mean":     g["auc"].mean(),
            "auc_std":      g["auc"].std(),
            "acc_mean":     g["acc"].mean(),
            "acc_std":      g["acc"].std(),
            "sens_mean":    g["sens"].mean(),
            "sens_std":     g["sens"].std(),
            "spec_mean":    g["spec"].mean(),
            "spec_std":     g["spec"].std(),
            "brier_mean":   g["brier"].mean(),
            "brier_std":    g["brier"].std(),
        })
    out = per_fold.groupby(CONFIG_COLS).apply(agg).reset_index()
    return out.sort_values("auc_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Threshold sweep — select on dev, apply to test
# ---------------------------------------------------------------------------

def threshold_sweep(test_df, dev_df):
    """
    For every (config, test_set, dev_set), compute a threshold on the dev
    fold using Youden's J, then apply that threshold to the test fold.
    Also report sens/spec at fixed test thresholds for comparison.
    """
    fixed_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Index dev predictions by (config, test_set, dev_set)
    dev_index = {}
    for key, g in dev_df.groupby(CONFIG_COLS + FOLD_COLS):
        dev_index[key] = (g["true_label"].values, g["prob_pos"].values)

    rows = []
    for key, g in test_df.groupby(CONFIG_COLS + FOLD_COLS):
        cfg = dict(zip(CONFIG_COLS + FOLD_COLS, key))
        y_true = g["true_label"].values
        y_prob = g["prob_pos"].values
        n = len(g)

        row = {**cfg, "n_patients": n}

        # Fixed thresholds on test
        for t in fixed_thresholds:
            sens, spec = sens_spec_at_threshold(y_true, y_prob, t)
            row[f"sens@{t}"] = sens
            row[f"spec@{t}"] = spec

        # Youden threshold from the paired dev fold
        if key in dev_index:
            dev_y, dev_p = dev_index[key]
            if len(np.unique(dev_y)) > 1:
                thr = youden_threshold(dev_y, dev_p)
                sens, spec = sens_spec_at_threshold(y_true, y_prob, thr)
                row["youden_thr"]     = thr
                row["youden_sens"]    = sens
                row["youden_spec"]    = spec

                thr90 = threshold_at_spec(dev_y, dev_p, 0.90)
                sens90, spec90 = sens_spec_at_threshold(y_true, y_prob, thr90)
                row["spec90_thr"]     = thr90
                row["spec90_sens"]    = sens90
                row["spec90_spec"]    = spec90

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-patient aggregation and error analysis
# ---------------------------------------------------------------------------

def per_patient_predictions(test_df):
    """
    For each (config, patient), average the model's probability across all
    test fold pairs the patient appears in. Each patient appears in exactly
    9 test fold pairs per config (one for each dev fold).
    """
    agg = (test_df
           .groupby(CONFIG_COLS + ["patient_id"])
           .agg(true_label=("true_label", "first"),
                prob_pos=("prob_pos", "mean"),
                n_appearances=("prob_pos", "count"))
           .reset_index())
    return agg


def error_analysis(per_patient, threshold=0.5):
    """
    Identify patients misclassified across the majority of their appearances.
    A patient appears in 9 test fold pairs per config.
    """
    pp = per_patient.copy()
    pp["pred_label"] = (pp["prob_pos"] >= threshold).astype(int)
    pp["correct"]    = (pp["pred_label"] == pp["true_label"]).astype(int)

    rows = []
    for cfg_key, g in pp.groupby(CONFIG_COLS):
        cfg = dict(zip(CONFIG_COLS, cfg_key))
        # In this per-patient view each patient is one row per config.
        # We want to know how often the patient was correct across the
        # underlying fold pairs.  Recount by joining back to the raw data.
        rows.append({**cfg,
                     "n_patients":    len(g),
                     "n_correct":     int(g["correct"].sum()),
                     "n_wrong":       int((1 - g["correct"]).sum())})

    return pd.DataFrame(rows)


def systematic_failures(test_df, threshold=0.5):
    """
    A patient is a systematic failure for a config if, across their 9 test
    fold pair appearances, they are misclassified in >= 7 of them.
    """
    pp = test_df.copy()
    pp["pred_label"] = (pp["prob_pos"] >= threshold).astype(int)
    pp["wrong"]      = (pp["pred_label"] != pp["true_label"]).astype(int)

    grouped = (pp
               .groupby(CONFIG_COLS + ["patient_id"])
               .agg(true_label=("true_label", "first"),
                    wrong_count=("wrong", "sum"),
                    n_runs=("wrong", "count"),
                    mean_prob=("prob_pos", "mean"))
               .reset_index())

    grouped["wrong_rate"] = grouped["wrong_count"] / grouped["n_runs"]
    systematic = grouped[grouped["wrong_rate"] >= 7 / 9].copy()
    return systematic.sort_values(CONFIG_COLS + ["wrong_rate"],
                                  ascending=[True] * len(CONFIG_COLS) + [False])


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def significance_tests(per_fold):
    """
    For each (fusion, arch, pretrained, aug) pair, compare the AUC across
    fold pairs against the corresponding `fusion=none` baseline.
    """
    # Restrict to test-set-level comparison: use the per-fold-pair AUCs.
    # Build the "baseline" AUC series for every non-fusion axis setting.
    rows = []

    base_mask = (
        (per_fold["fusion"] == "none") &
        (per_fold["arch"] == "resnet")
    )
    baseline = per_fold[base_mask]

    for (fusion, arch, pretrained, aug), g in per_fold.groupby(
        ["fusion", "arch", "pretrained", "aug"]
    ):
        if fusion == "none" and arch == "resnet":
            continue

        # Match the same (pretrained, aug, test_set, dev_set) rows
        candidate = g
        base_sel = baseline[
            (baseline["pretrained"] == pretrained) &
            (baseline["aug"] == aug)
        ]
        merged = candidate.merge(
            base_sel,
            on=["test_set", "dev_set"],
            suffixes=("_cand", "_base"),
        )
        if len(merged) < 5:
            continue

        a = merged["auc_cand"].values
        b = merged["auc_base"].values
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < 5:
            continue

        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, np.nan

        rows.append({
            "fusion":      fusion,
            "arch":        arch,
            "pretrained":  pretrained,
            "aug":         aug,
            "n_pairs":     len(a),
            "auc_cand":    a.mean(),
            "auc_base":    b.mean(),
            "delta":       a.mean() - b.mean(),
            "wilcoxon_p":  p,
        })

    return pd.DataFrame(rows).sort_values("delta", ascending=False)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_table(test_df, n_bins=10):
    """Reliability diagram bins, aggregated over all fold pairs per config."""
    rows = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for cfg_key, g in test_df.groupby(CONFIG_COLS):
        cfg = dict(zip(CONFIG_COLS, cfg_key))
        y_true = g["true_label"].values
        y_prob = g["prob_pos"].values
        bins   = np.digitize(y_prob, bin_edges[1:-1])
        for b in range(n_bins):
            mask = bins == b
            if not mask.any():
                continue
            rows.append({
                **cfg,
                "bin":              b,
                "bin_lower":        bin_edges[b],
                "bin_upper":        bin_edges[b + 1],
                "n":                int(mask.sum()),
                "mean_prob":        y_prob[mask].mean(),
                "empirical_rate":   y_true[mask].mean(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR)
    parser.add_argument("--output-dir",      default=OUTPUT_DIR)
    parser.add_argument("--threshold",       type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading predictions ...")
    raw = load_predictions(args.predictions_dir)
    dev_df  = raw[raw["set"] == "dev"].copy()
    test_df = filter_test(raw)

    print(f"  dev rows:  {len(dev_df):,}")
    print(f"  test rows: {len(test_df):,}")

    # --- per-fold-pair metrics --------------------------------------------
    print("\nComputing per-fold-pair metrics ...")
    per_fold = per_fold_pair_metrics(test_df)
    per_fold.to_csv(os.path.join(args.output_dir, "per_fold_pair.csv"), index=False)
    print(f"  wrote per_fold_pair.csv ({len(per_fold)} rows)")

    # --- per-config summary ------------------------------------------------
    print("\nAggregating to per-config summary ...")
    summary = per_config_summary(per_fold)
    summary.to_csv(os.path.join(args.output_dir, "per_config_summary.csv"), index=False)
    print(f"  wrote per_config_summary.csv ({len(summary)} configs)")

    print("\n" + "=" * 110)
    print("PER-CONFIG TEST AUC (mean ± std across 90 fold pairs)")
    print("=" * 110)
    cols = ["fusion", "arch", "pretrained", "aug", "lr",
            "auc_mean", "auc_std", "acc_mean", "sens_mean", "spec_mean", "folds"]
    print(summary[cols].to_string(index=False))

    # --- threshold sweep ---------------------------------------------------
    print("\nComputing threshold sweep ...")
    sweep = threshold_sweep(test_df, dev_df)
    sweep.to_csv(os.path.join(args.output_dir, "threshold_sweep.csv"), index=False)
    print(f"  wrote threshold_sweep.csv ({len(sweep)} rows)")

    # --- per-patient aggregation ------------------------------------------
    print("\nAggregating per-patient predictions ...")
    pp = per_patient_predictions(test_df)
    pp.to_csv(os.path.join(args.output_dir, "per_patient.csv"), index=False)
    print(f"  wrote per_patient.csv ({len(pp)} rows)")

    # --- systematic failures ----------------------------------------------
    print("\nIdentifying systematic failures ...")
    fails = systematic_failures(test_df, threshold=args.threshold)
    fails.to_csv(os.path.join(args.output_dir, "error_analysis.csv"), index=False)
    print(f"  wrote error_analysis.csv ({len(fails)} rows)")

    # --- significance -----------------------------------------------------
    print("\nRunning significance tests ...")
    sig = significance_tests(per_fold)
    sig.to_csv(os.path.join(args.output_dir, "significance_tests.csv"), index=False)
    if not sig.empty:
        print("  wrote significance_tests.csv")
        print("\n" + "=" * 90)
        print("FUSION VS BASELINE (Wilcoxon signed-rank on paired fold-pair AUCs)")
        print("=" * 90)
        print(sig[["fusion", "arch", "pretrained", "aug", "n_pairs",
                   "auc_cand", "auc_base", "delta", "wilcoxon_p"]].to_string(index=False))

    # --- calibration ------------------------------------------------------
    print("\nComputing calibration table ...")
    calib = calibration_table(test_df)
    calib.to_csv(os.path.join(args.output_dir, "calibration.csv"), index=False)
    print(f"  wrote calibration.csv ({len(calib)} bins)")

    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()