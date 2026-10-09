
"""
variant_speech_only.py

Trains a ResNet-18 on the speech spectrogram alone, using the same input
pipeline as the single-modality cough baseline.

The input is the patient-level speech tensor stored as a .pt file:
    [128, T]  ->  used directly as a spectrogram
    [128]     ->  broadcast along the time axis as a fallback

Tests: does the speech modality carry TB signal on its own?

Usage:
    python3 src/variant_speech_only.py --fold 0 1
    python3 src/variant_speech_only.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dataloader import apply_augmentation
from utils import (
    pad_to_224, min_max_rescale, imagenet_normalize,
    seed_worker, create_generator, standard_collate,
)
from model_scripts import ResNet18, train_validate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDS   = "data/cage/data_folds_filtered"
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


class SpeechOnlyDataset(Dataset):
    """One sample per patient. The speech tensor is used as a spectrogram
    image, exactly like the cough baseline."""

    def __init__(self, annotations_file, speech_dir,
                 is_train=False, augmentation="none"):
        self.df = pd.read_csv(annotations_file)
        self.df["patient_id"] = self.df["Cough_ID"].astype(str).apply(
            lambda x: x.split("/")[0])
        self.patients = (self.df.groupby("patient_id")
                         .agg({"Status": "first"})
                         .reset_index())
        self.patients = self.patients[self.patients["patient_id"].map(
            lambda pid: os.path.exists(os.path.join(speech_dir, f"{pid}.pt"))
        )].reset_index(drop=True)

        # Train_validate reads .samples to compute class weights.
        self.samples = [
            (str(r["patient_id"]), None, int(r["Status"]))
            for _, r in self.patients.iterrows()
        ]
        self.speech_dir   = speech_dir
        self.is_train     = is_train
        self.augmentation = augmentation

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        row = self.patients.iloc[idx]
        pid   = str(row["patient_id"])
        label = int(row["Status"])

        s_raw = torch.load(os.path.join(self.speech_dir, f"{pid}.pt"),
                           weights_only=True)

        # Use the speech tensor as a spectrogram, same as a cough.
        if s_raw.ndim == 2:
            s_img = s_raw                                          # [128, T]
        else:
            # Fallback for 1-D tensors already averaged over time.
            s_img = s_raw.unsqueeze(1).repeat(1, 43)               # [128, 43]

        s_img = pad_to_224(s_img)                                  # [224, 224]
        s_img = min_max_rescale(s_img)                             # [0, 1]

        if self.is_train and self.augmentation != "none":
            s_img = apply_augmentation(s_img, self.augmentation)

        s_3ch = s_img.unsqueeze(0).repeat(3, 1, 1)                 # [3, 224, 224]
        s_3ch = imagenet_normalize(s_3ch)                          # ImageNet stats
        return s_3ch, label, pid


def get_speech_only_data(test_fold, dev_fold, batch_size, augmentation="none"):
    train_folds = [
        f"{DATA_FOLDS}/fold_{k}.csv"
        for k in range(10) if k != dev_fold and k != test_fold
    ]
    train_ds = ConcatDataset([
        SpeechOnlyDataset(f, SPEECH_DIR, is_train=True, augmentation=augmentation)
        for f in train_folds
    ])
    val_ds  = SpeechOnlyDataset(f"{DATA_FOLDS}/fold_{dev_fold}.csv",
                                SPEECH_DIR, is_train=False,
                                augmentation=augmentation)
    test_ds = SpeechOnlyDataset(f"{DATA_FOLDS}/fold_{test_fold}.csv",
                                SPEECH_DIR, is_train=False,
                                augmentation=augmentation)

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
    parser.add_argument("--fold", type=int, nargs=2, default=None,
                        metavar=("TEST", "DEV"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir  = os.path.join(PROJECT_ROOT, "results_variant_speech_only")
    pred_dir = os.path.join(PROJECT_ROOT, "predictions_variant_speech_only")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    if args.fold is not None:
        pairs = [tuple(args.fold)]
    else:
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        if args.limit:
            pairs = pairs[:args.limit]

    print(f"Variant: speech-only baseline (speech used as spectrogram image)")
    print(f"Fold pairs: {len(pairs)}\n")

    rows = []
    for n, (test_fold, dev_fold) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] test={test_fold} dev={dev_fold} ... ",
              end="", flush=True)
        try:
            train, val, test = get_speech_only_data(
                test_fold, dev_fold, CFG["batch_size"],
                augmentation=CFG["aug"])

            model = ResNet18(num_classes=2,
                             use_pretrained=CFG["use_pretrained"]).to(device)

            params = {
                "fusion": "speech_only", "arch": "resnet",
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
    print("SPEECH-ONLY SUMMARY")
    print("=" * 60)
    print(summary.to_string())


if __name__ == "__main__":
    main()