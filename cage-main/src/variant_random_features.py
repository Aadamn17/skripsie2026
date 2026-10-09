"""
variant_random_features.py

Frozen randomly-initialised ResNet-18 with only the classifier head trained.
Quantifies how much of the performance comes from ImageNet pretraining vs
from the architecture alone.

Usage:
    python3 src/variant_random_features.py --fold 0 1
    python3 src/variant_random_features.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

from dataloader import get_data
from model_scripts import train_validate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDS   = "data/cage/data_folds_filtered"
COUGH_DIR    = "data/cage/mel_spectrograms_128"

CFG = dict(lr=1e-5, wd=1e-4, aug="none", batch_size=32, num_epochs=50,
           es_patience=5, es_min_delta=5e-3, es_min_epochs=15, es_window=5)


class RandomFeatureResNet18(nn.Module):
    """Random-init ResNet-18. All conv/linear weights frozen; BN updates; fc trained."""

    def __init__(self, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.resnet = resnet18(weights=None)  # random init
        # Freeze everything except fc
        for name, p in self.resnet.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False
        # Replace fc; dropout before classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )
        # fc parameters are trainable by default
        for p in self.resnet.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs=2, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir  = os.path.join(PROJECT_ROOT, "results_variant_random_features")
    pred_dir = os.path.join(PROJECT_ROOT, "predictions_variant_random_features")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    if args.fold is not None:
        pairs = [tuple(args.fold)]
    else:
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        if args.limit:
            pairs = pairs[:args.limit]

    print(f"Variant: random-features baseline\nFold pairs: {len(pairs)}\n")

    rows = []
    for n, (test_fold, dev_fold) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] test={test_fold} dev={dev_fold} ... ",
              end="", flush=True)
        try:
            train, val, test = get_data(
                dataset="cage", data_folds=DATA_FOLDS,
                i=test_fold, j=dev_fold, cough_dir=COUGH_DIR,
                loss="cross_entropy_resnet", batch_size=CFG["batch_size"],
                augmentation=CFG["aug"])

            model = RandomFeatureResNet18().to(device)

            params = {
                "fusion": "random_features", "arch": "resnet",
                "learning_rate": CFG["lr"], "weight_decay": CFG["wd"],
                "augmentation": CFG["aug"], "use_pretrained": False,
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
                print("no best"); continue
            b = best[0]
            rows.append({"test_set": test_fold, "dev_set": dev_fold,
                         "best_epoch": b["epoch"],
                         "test_auc": b["test_auc"], "test_acc": b["test_acc"],
                         "test_sens": b["test_sens"], "test_spec": b["test_spec"]})
            print(f"auc={b['test_auc']:.4f}  epoch={b['epoch']}")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    summary = pd.DataFrame({
        "mean": df[["test_auc","test_acc","test_sens","test_spec"]].mean().round(4),
        "std":  df[["test_auc","test_acc","test_sens","test_spec"]].std().round(4),
    })
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    print("\n" + "=" * 60)
    print("RANDOM-FEATURES SUMMARY")
    print("=" * 60)
    print(summary.to_string())


if __name__ == "__main__":
    main()