
"""
variant_interaction_fusion.py

Early fusion with three channels: [cough, speech, cough*speech].

The interaction channel is the element-wise product of the cough and
speech channels, both of which are in [0, 1] after min-max rescaling,
so the product also lies in [0, 1] and can be ImageNet-normalised with
the other two channels.

Writes to separate output folders so the main grid's results are not
touched.  Reuses ResNet18 and train_validate from model_scripts without
modification.

Usage:
    python3 src/variant_interaction_fusion.py --fold 0 1
    python3 src/variant_interaction_fusion.py --limit 5
    python3 src/variant_interaction_fusion.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from dataloader import EarlyFusionFlatDataset, apply_augmentation
from utils import (
    pad_to_224, min_max_rescale, imagenet_normalize,
    seed_worker, create_generator, standard_collate,
)
from model_scripts import ResNet18, train_validate


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


class InteractionFusionDataset(EarlyFusionFlatDataset):
    """Three channels: cough, speech, cough*speech."""

    def __getitem__(self, idx):
        cid, pid, label = self.samples[idx]
        c_raw = torch.tensor(
            np.load(os.path.join(self.cough_dir, cid + ".npy")),
            dtype=torch.float32,
        )

        # Cough channel
        c_img = pad_to_224(c_raw)
        c_img = min_max_rescale(c_img)
        if self.is_train and self.augmentation != "none":
            c_img = apply_augmentation(c_img, self.augmentation)

        # Speech channel
        m_speech = self._mean_speech_image(pid)
        if m_speech.ndim == 2:
            m_speech = m_speech.mean(dim=1)
        m_speech = m_speech.unsqueeze(1).repeat(1, 43)
        s_img = pad_to_224(m_speech)
        s_img = min_max_rescale(s_img)

        # Interaction channel: element-wise product
        # Both operands are in [0, 1], so the product is also in [0, 1]
        interaction = c_img * s_img

        fused = torch.stack([c_img, s_img, interaction], dim=0)
        fused = imagenet_normalize(fused)
        return fused, label, pid


def get_variant_data(test_fold, dev_fold, batch_size):
    train_folds = [
        f"{DATA_FOLDS}/fold_{k}.csv"
        for k in range(10) if k != dev_fold and k != test_fold
    ]
    train_ds = ConcatDataset([
        InteractionFusionDataset(f, COUGH_DIR, SPEECH_DIR, arch="resnet",
                                 is_train=True, augmentation=CFG["aug"])
        for f in train_folds
    ])
    val_ds  = InteractionFusionDataset(
        f"{DATA_FOLDS}/fold_{dev_fold}.csv", COUGH_DIR, SPEECH_DIR,
        arch="resnet", is_train=False, augmentation=CFG["aug"])
    test_ds = InteractionFusionDataset(
        f"{DATA_FOLDS}/fold_{test_fold}.csv", COUGH_DIR, SPEECH_DIR,
        arch="resnet", is_train=False, augmentation=CFG["aug"])

    g = create_generator()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True,
                              collate_fn=standard_collate,
                              worker_init_fn=seed_worker, generator=g)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=standard_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=standard_collate)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fold", type=int, nargs=2, default=None,
                        metavar=("TEST", "DEV"))
    args = parser.parse_args()

    out_dir  = os.path.join(PROJECT_ROOT, "results_variant_interaction")
    pred_dir = os.path.join(PROJECT_ROOT, "predictions_variant_interaction")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    if args.fold is not None:
        pairs = [tuple(args.fold)]
    else:
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        if args.limit:
            pairs = pairs[:args.limit]

    print(f"Variant: [cough, speech, cough*speech]")
    print(f"Fold pairs: {len(pairs)}")
    print(f"Output: {out_dir}")
    print()

    rows = []
    for n, (test_fold, dev_fold) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] test={test_fold} dev={dev_fold} ... ",
              end="", flush=True)

        try:
            train, val, test = get_variant_data(
                test_fold, dev_fold, CFG["batch_size"])

            model = ResNet18(
                num_classes=2, use_pretrained=CFG["use_pretrained"]).to(device)

            params = {
                "fusion":           "early_interaction",
                "arch":             "resnet",
                "learning_rate":    CFG["lr"],
                "weight_decay":     CFG["wd"],
                "augmentation":     CFG["aug"],
                "use_pretrained":   CFG["use_pretrained"],
                "test_set":         test_fold,
                "dev_set":          dev_fold,
                "num_epochs":       CFG["num_epochs"],
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
                print("no best row")
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
    print("VARIANT SUMMARY: [cough, speech, cough*speech]")
    print("=" * 60)
    print(summary.to_string())
    print(f"\nFold pairs completed: {len(df)} / {len(pairs)}")


if __name__ == "__main__":
    main()