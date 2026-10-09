"""
variant_backbone.py

Tests three backbone/freeze-policy variants at the best config
(lr=1e-5, wd=1e-4, aug=none, pretrained=True):

    resnet_full_freeze   ResNet-18, only the classifier head trainable
    resnet_layer34       ResNet-18, layer3 + layer4 + head trainable
    efficientnet_b0      EfficientNet-B0, last feature blocks + head trainable

Writes to isolated output folders so the main grid is untouched.

Usage:
    python3 src/variant_backbone.py --variant resnet_full_freeze --fold 0 1
    python3 src/variant_backbone.py --variant resnet_full_freeze
    python3 src/variant_backbone.py --variant resnet_layer34
    python3 src/variant_backbone.py --variant efficientnet_b0
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

from dataloader import get_data
from model_scripts import train_validate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDS   = "data/cage/data_folds_filtered"
COUGH_DIR    = "data/cage/mel_spectrograms_128"

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


# ---------------------------------------------------------------------------
# Backbone wrappers with configurable freeze policies
# ---------------------------------------------------------------------------

class ResNet18Policy(nn.Module):
    """ResNet-18 with ImageNet weights, configurable unfreeze policy."""

    def __init__(self, num_classes=2, dropout_p=0.3, use_pretrained=True,
                 unfreeze_prefixes=("layer4",)):
        super().__init__()
        self.use_pretrained = use_pretrained
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.resnet = resnet18(weights=weights)

        if use_pretrained:
            for p in self.resnet.parameters():
                p.requires_grad = False
            for name, p in self.resnet.named_parameters():
                if name.startswith("fc"):
                    continue
                for pfx in unfreeze_prefixes:
                    if name.startswith(pfx):
                        p.requires_grad = True
                        break

        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.use_pretrained and mode:
            for m in self.resnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        return self.resnet(x)


class EfficientNetB0Policy(nn.Module):
    """EfficientNet-B0 with ImageNet weights, last feature blocks trainable."""

    def __init__(self, num_classes=2, dropout_p=0.3, use_pretrained=True,
                 unfreeze_prefixes=("features.6", "features.7", "features.8")):
        super().__init__()
        self.use_pretrained = use_pretrained
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.eff = efficientnet_b0(weights=weights)

        if use_pretrained:
            for p in self.eff.parameters():
                p.requires_grad = False
            for name, p in self.eff.named_parameters():
                if name.startswith("classifier"):
                    continue
                for pfx in unfreeze_prefixes:
                    if name.startswith(pfx):
                        p.requires_grad = True
                        break

        self.eff.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(1280, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.use_pretrained and mode:
            for m in self.eff.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        return self.eff(x)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "resnet_full_freeze": dict(
        builder = lambda: ResNet18Policy(unfreeze_prefixes=()),
        out_dir = "results_variant_resnet_full_freeze",
        pred_dir = "predictions_variant_resnet_full_freeze",
        name    = "ResNet-18 (full freeze)",
        arch    = "resnet",
    ),
    "resnet_layer34": dict(
        builder = lambda: ResNet18Policy(unfreeze_prefixes=("layer3", "layer4")),
        out_dir = "results_variant_resnet_layer34",
        pred_dir = "predictions_variant_resnet_layer34",
        name    = "ResNet-18 (layer3 + layer4)",
        arch    = "resnet",
    ),
    "efficientnet_b0": dict(
        builder = lambda: EfficientNetB0Policy(),
        out_dir = "results_variant_efficientnet_b0",
        pred_dir = "predictions_variant_efficientnet_b0",
        name    = "EfficientNet-B0 (last blocks)",
        arch    = "efficientnet",
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N fold pairs (for testing).")
    parser.add_argument("--fold", type=int, nargs=2, default=None,
                        metavar=("TEST", "DEV"),
                        help="Run a single fold pair.")
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    out_dir  = os.path.join(PROJECT_ROOT, spec["out_dir"])
    pred_dir = os.path.join(PROJECT_ROOT, spec["pred_dir"])
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    if args.fold is not None:
        pairs = [tuple(args.fold)]
    else:
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        if args.limit:
            pairs = pairs[:args.limit]

    print(f"Variant: {spec['name']}")
    print(f"Fold pairs: {len(pairs)}")
    print(f"Output: {out_dir}")
    print(f"Predictions: {pred_dir}")
    print()

    rows = []
    for n, (test_fold, dev_fold) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] test={test_fold} dev={dev_fold} ... ",
              end="", flush=True)

        try:
            train, val, test = get_data(
                dataset      = "cage",
                data_folds   = DATA_FOLDS,
                i            = test_fold,
                j            = dev_fold,
                cough_dir    = COUGH_DIR,
                loss         = "cross_entropy_resnet",
                batch_size   = CFG["batch_size"],
                augmentation = CFG["aug"],
            )

            model = spec["builder"]().to(device)

            params = {
                "fusion":                args.variant,
                "arch":                  spec["arch"],
                "learning_rate":         CFG["lr"],
                "weight_decay":          CFG["wd"],
                "augmentation":          CFG["aug"],
                "use_pretrained":        CFG["use_pretrained"],
                "test_set":              test_fold,
                "dev_set":               dev_fold,
                "num_epochs":            CFG["num_epochs"],
                "early_stop_patience":   CFG["es_patience"],
                "early_stop_min_delta":  CFG["es_min_delta"],
                "early_stop_min_epochs": CFG["es_min_epochs"],
                "early_stop_window":     CFG["es_window"],
            }

            history = train_validate(train, val, test, model, params,
                                     predictions_dir=pred_dir)

            best = [h for h in history if h["is_best"] and
                    not np.isnan(h.get("test_auc", float("nan")))]
            if not best:
                print("no best row (all test AUC are NaN)")
                continue

            b = best[0]
            rows.append({
                "test_set":   test_fold,
                "dev_set":    dev_fold,
                "best_epoch": b["epoch"],
                "test_auc":   b["test_auc"],
                "test_acc":   b["test_acc"],
                "test_sens":  b["test_sens"],
                "test_spec":  b["test_spec"],
            })
            print(f"auc={b['test_auc']:.4f}  epoch={b['epoch']}")

        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    if not rows:
        print("No successful runs.")
        return

    df = pd.DataFrame(rows)
    per_fold_path = os.path.join(out_dir, "per_fold.csv")
    df.to_csv(per_fold_path, index=False)

    summary = pd.DataFrame({
        "mean": df[["test_auc", "test_acc", "test_sens", "test_spec"]].mean().round(4),
        "std":  df[["test_auc", "test_acc", "test_sens", "test_spec"]].std().round(4),
    })
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path)

    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {summary_path}")
    print("\n" + "=" * 60)
    print(f"VARIANT SUMMARY: {spec['name']}")
    print("=" * 60)
    print(summary.to_string())
    print(f"\nFold pairs completed: {len(df)} / {len(pairs)}")


if __name__ == "__main__":
    main()
