"""
learning_curves.py

Trains the best config (late fusion, resnet, lr=1e-5, aug=none, pretrained)
on progressively larger fractions of the training set: 25%, 50%, 75%, 100%.

Produces a learning curve of test AUC vs training set size, with error bars.

Usage:
    python3 src/learning_curves.py --frac 0.25 --fold 0 1
    python3 src/learning_curves.py --frac 0.25
    python3 src/learning_curves.py               # all four fractions
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dataloader import LateFusionDataset, late_fusion_collate
from utils import seed_worker, create_generator
from model_scripts import LateFusion, train_validate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDS   = "data/cage/data_folds_filtered"
COUGH_DIR    = "data/cage/mel_spectrograms_128"
SPEECH_DIR   = "data/cage/preprocessed_speech"

CFG = dict(
    lr              = 1e-5,
    wd              = 1e-4,
    aug             = "none",
    use_pretrained  = True,
    batch_size      = 32,
    num_epochs      = 50,
    es_patience     = 5,
    es_min_delta    = 5e-3,
    es_min_epochs   = 15,
    es_window       = 5,
)


def get_subset_data(test_fold, dev_fold, frac, batch_size, seed=42):
    """Build train/val/test loaders with a random `frac` subset of training samples."""
    train_folds = [
        f"{DATA_FOLDS}/fold_{k}.csv"
        for k in range(10) if k != dev_fold and k != test_fold
    ]
    full_train = ConcatDataset([
        LateFusionDataset(f, COUGH_DIR, SPEECH_DIR,
                          is_train=True, augmentation=CFG["aug"])
        for f in train_folds
    ])

    n_total = len(full_train)
    n_keep  = max(1, int(round(frac * n_total)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_keep, replace=False).tolist()
    train_ds = Subset(full_train, idx)

    val_ds  = LateFusionDataset(f"{DATA_FOLDS}/fold_{dev_fold}.csv",
                                COUGH_DIR, SPEECH_DIR,
                                is_train=False, augmentation=CFG["aug"])
    test_ds = LateFusionDataset(f"{DATA_FOLDS}/fold_{test_fold}.csv",
                                COUGH_DIR, SPEECH_DIR,
                                is_train=False, augmentation=CFG["aug"])

    g = create_generator()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True,
                              collate_fn=late_fusion_collate,
                              worker_init_fn=seed_worker, generator=g)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=late_fusion_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=late_fusion_collate)
    return train_loader, val_loader, test_loader, n_keep, n_total


def train_one(test_fold, dev_fold, frac, pred_dir):
    train, val, test, n_keep, n_total = get_subset_data(
        test_fold, dev_fold, frac, CFG["batch_size"])

    model = LateFusion(num_classes=2,
                       use_pretrained=CFG["use_pretrained"]).to(device)

    params = {
        "fusion": "late", "arch": "resnet",
        "learning_rate": CFG["lr"], "weight_decay": CFG["wd"],
        "augmentation": CFG["aug"], "use_pretrained": CFG["use_pretrained"],
        "test_set": test_fold, "dev_set": dev_fold,
        "num_epochs": CFG["num_epochs"],
        "early_stop_patience": CFG["es_patience"],
        "early_stop_min_delta": CFG["es_min_delta"],
        "early_stop_min_epochs": CFG["es_min_epochs"],
        "early_stop_window": CFG["es_window"],
    }
    history = train_validate(train, val, test, model, params,
                             predictions_dir=pred_dir)
    best = [h for h in history if h["is_best"] and
            not np.isnan(h.get("test_auc", float("nan")))]
    if not best:
        return None
    b = best[0]
    return {
        "frac": frac, "n_train": n_keep, "n_total": n_total,
        "test_set": test_fold, "dev_set": dev_fold,
        "best_epoch": b["epoch"],
        "test_auc": b["test_auc"], "test_acc": b["test_acc"],
        "test_sens": b["test_sens"], "test_spec": b["test_spec"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=None,
                        help="Single fraction (0-1). If omitted, runs all four.")
    parser.add_argument("--fold", type=int, nargs=2, default=None,
                        metavar=("TEST", "DEV"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir  = os.path.join(PROJECT_ROOT, "results_learning_curves")
    pred_dir = os.path.join(PROJECT_ROOT, "predictions_learning_curves")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    if args.fold is not None:
        pairs = [tuple(args.fold)]
    else:
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        if args.limit:
            pairs = pairs[:args.limit]

    fracs = [args.frac] if args.frac is not None else [0.25, 0.50, 0.75, 1.00]
    print(f"Fractions: {fracs}")
    print(f"Fold pairs per fraction: {len(pairs)}\n")

    all_rows = []
    for frac in fracs:
        print(f"\n=== Fraction {frac:.2f} ===")
        for n, (test_fold, dev_fold) in enumerate(pairs, 1):
            print(f"[{n}/{len(pairs)}] test={test_fold} dev={dev_fold} ... ",
                  end="", flush=True)
            try:
                row = train_one(test_fold, dev_fold, frac, pred_dir)
                if row is None:
                    print("no best"); continue
                all_rows.append(row)
                print(f"auc={row['test_auc']:.4f}  n_train={row['n_train']}")
            except Exception as exc:
                print(f"FAILED: {type(exc).__name__}: {exc}")

    if not all_rows:
        print("No successful runs.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)

    summary = df.groupby("frac").agg(
        n_train=("n_train", "first"),
        auc_mean=("test_auc", "mean"),
        auc_std=("test_auc", "std"),
        acc_mean=("test_acc", "mean"),
        sens_mean=("test_sens", "mean"),
        spec_mean=("test_spec", "mean"),
        n_folds=("test_auc", "count"),
    ).round(4).reset_index()
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("LEARNING CURVE SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()