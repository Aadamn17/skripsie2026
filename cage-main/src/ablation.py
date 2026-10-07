"""
ablation_study.py

Trains a LateFusion model for each (test, dev) fold pair, then evaluates the
trained model under four conditions at inference time:

    intact   - real patient speech
    zero     - zeros in place of speech
    shuffle  - speech shuffled within the batch
    cough    - time-averaged cough of the same event, used as pseudo-speech

The last condition distinguishes "the model uses the second branch" from
"the model uses speech specifically." If the model performs the same under
`intact` and `cough`, the second branch is redundant with the cough. If it
performs better under `intact`, the speech representation carries information
that the cough branch does not.

Outputs:
    src/analysis/ablation_per_fold.csv     one row per fold pair
    src/analysis/ablation_summary.csv      mean ± std per condition
    figures/ablation_boxplot.pdf           boxplot of the four AUC distributions
    figures/ablation_deltas.pdf            bar chart of the three deltas
    figures/ablation_boxplot.png           same, quick preview
    figures/ablation_deltas.png            same, quick preview

Usage:
    python3 src/ablation_study.py --fold 0 1
    python3 src/ablation_study.py --limit 5
    python3 src/ablation_study.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, ConcatDataset
from sklearn import metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import LateFusionDataset
from utils import (
    pad_to_224, min_max_rescale, imagenet_normalize,
    seed_worker, create_generator,
)
from model_scripts import LateFusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP    = dict(device_type="cuda", dtype=torch.bfloat16)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "analysis")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "figures")

DATA_FOLDS = "data/cage/data_folds_filtered"
COUGH_DIR  = "data/cage/mel_spectrograms_128"
SPEECH_DIR = "data/cage/preprocessed_speech"

# Best configuration from the grid search
CFG = dict(
    lr              = 1e-5,
    wd              = 1e-4,
    batch_size      = 32,
    num_epochs      = 50,
    use_pretrained  = True,
    augmentation    = "none",
    es_patience     = 5,
    es_min_delta    = 5e-3,
    es_min_epochs   = 15,
    es_window       = 5,
)


# ---------------------------------------------------------------------------
# Extended dataset that also returns a time-averaged cough vector
# ---------------------------------------------------------------------------

class AblationLateFusionDataset(LateFusionDataset):
    """Same as LateFusionDataset but yields an additional 128-dim cough vector."""

    def __getitem__(self, idx):
        c_3ch, speech, label, pid = super().__getitem__(idx)

        row  = self.df.iloc[idx]
        cid  = str(row["Cough_ID"])
        c_raw = torch.tensor(
            np.load(os.path.join(self.cough_dir, cid + ".npy")),
            dtype=torch.float32,
        )
        if c_raw.ndim == 2:
            cough_vec = c_raw.mean(dim=1)
        else:
            cough_vec = c_raw
        cough_vec = (cough_vec - cough_vec.mean()) / (cough_vec.std() + 1e-8)

        return c_3ch, speech, cough_vec, label, pid


def ablation_collate(batch):
    cough, speech, cough_vec, labels, pids = zip(*batch)
    return (torch.stack(cough),
            torch.stack(speech),
            torch.stack(cough_vec),
            torch.tensor(labels, dtype=torch.long),
            list(pids))


def get_ablation_data(test_fold, dev_fold, batch_size):
    """Return (train_loader, dev_loader, test_loader) for one fold pair."""
    train_folds = [
        f"{DATA_FOLDS}/fold_{k}.csv"
        for k in range(10) if k != dev_fold and k != test_fold
    ]

    train_ds = ConcatDataset([
        AblationLateFusionDataset(f, COUGH_DIR, SPEECH_DIR,
                                  is_train=True, augmentation=CFG["augmentation"])
        for f in train_folds
    ])
    dev_ds  = AblationLateFusionDataset(
        f"{DATA_FOLDS}/fold_{dev_fold}.csv", COUGH_DIR, SPEECH_DIR,
        is_train=False, augmentation=CFG["augmentation"])
    test_ds = AblationLateFusionDataset(
        f"{DATA_FOLDS}/fold_{test_fold}.csv", COUGH_DIR, SPEECH_DIR,
        is_train=False, augmentation=CFG["augmentation"])

    g = create_generator()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        collate_fn=ablation_collate,
        worker_init_fn=seed_worker, generator=g,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
        collate_fn=ablation_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
        collate_fn=ablation_collate,
    )
    return train_loader, dev_loader, test_loader


# ---------------------------------------------------------------------------
# Early stopper (same logic as model_scripts.EarlyStopper)
# ---------------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience, min_delta, window, min_epochs):
        self.patience, self.min_delta = patience, min_delta
        self.window, self.min_epochs  = window, min_epochs
        self.history = []
        self.best = -float("inf")
        self.counter = 0
        self.best_state = None
        self.best_epoch = 0
        self.should_stop = False

    def step(self, value, model, epoch):
        if value is None or (isinstance(value, float) and value != value):
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.should_stop = True
            return False
        self.history.append(value)
        if len(self.history) < self.window:
            return False
        smoothed = sum(self.history[-self.window:]) / self.window
        if smoothed > self.best + self.min_delta:
            self.best = smoothed
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        if self.counter >= self.patience and epoch >= self.min_epochs:
            self.should_stop = True
        return False

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Ablated forward pass
# ---------------------------------------------------------------------------

def forward_ablated(model, cough, speech, cough_vec, mode, shuffle_seed=42):
    """Reproduce LateFusion.forward with the second-branch input replaced
    according to `mode`."""
    cough_logits = model.cough_backbone(cough)

    if mode == "intact":
        second = speech
    elif mode == "zero":
        second = torch.zeros_like(speech)
    elif mode == "shuffle":
        g = torch.Generator(device=speech.device).manual_seed(shuffle_seed)
        idx = torch.randperm(speech.size(0), generator=g,
                             device=speech.device)
        second = speech[idx]
    elif mode == "cough":
        second = cough_vec
    else:
        raise ValueError(f"Unknown mode: {mode}")

    speech_logits = model.speech_lr(second)
    fused = torch.cat([cough_logits, speech_logits], dim=1)
    return model.fusion_head(fused)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def aggregate_patient(logits_by_patient, labels_by_patient):
    pids = list(logits_by_patient.keys())
    agg_logits = torch.stack(
        [torch.stack(logits_by_patient[pid]).mean(0) for pid in pids]
    )
    y_true = np.array([labels_by_patient[pid] for pid in pids])
    y_prob = torch.softmax(agg_logits, dim=1)[:, 1].numpy()
    return y_true, y_prob


def auc_or_nan(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return metrics.roc_auc_score(y_true, y_prob)


def evaluate_in_mode(model, loader, mode, shuffle_seed=42):
    model.eval()
    logits_by_patient, labels_by_patient = {}, {}
    with torch.inference_mode():
        for cough, speech, cough_vec, labels, pids in loader:
            cough      = cough.to(torch.float32).to(device)
            speech     = speech.to(torch.float32).to(device)
            cough_vec  = cough_vec.to(torch.float32).to(device)
            with autocast(**AMP):
                logits = forward_ablated(model, cough, speech, cough_vec,
                                         mode, shuffle_seed)
            logits = logits.float().cpu()
            for i, pid in enumerate(pids):
                logits_by_patient.setdefault(pid, []).append(logits[i])
                labels_by_patient.setdefault(pid, int(labels[i]))
    y_true, y_prob = aggregate_patient(logits_by_patient, labels_by_patient)
    return auc_or_nan(y_true, y_prob)


# ---------------------------------------------------------------------------
# Training + ablation for one fold pair
# ---------------------------------------------------------------------------

def class_weights_from_loader(train_loader):
    counts = torch.zeros(2, dtype=torch.long)
    for _, _, _, labels, _ in train_loader:
        for l in labels:
            counts[int(l)] += 1
    counts = counts.clamp(min=1)
    return counts.sum() / (2 * counts.float())


def train_and_ablate(test_fold, dev_fold):
    print(f"[fold test={test_fold} dev={dev_fold}] loading data ...")
    train_loader, dev_loader, test_loader = get_ablation_data(
        test_fold, dev_fold, CFG["batch_size"],
    )

    class_weights = class_weights_from_loader(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = LateFusion(
        num_classes=2,
        use_pretrained=CFG["use_pretrained"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.param_groups(CFG["lr"], CFG["wd"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["num_epochs"]
    )
    stopper = EarlyStopper(CFG["es_patience"], CFG["es_min_delta"],
                           CFG["es_window"], CFG["es_min_epochs"])

    # --- training ---
    for epoch in range(CFG["num_epochs"]):
        model.train()
        for cough, speech, _, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            cough  = cough.to(torch.float32).to(device)
            speech = speech.to(torch.float32).to(device)
            with autocast(**AMP):
                out  = model(cough, speech)
                loss = criterion(out, labels.long().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        dev_auc = evaluate_in_mode(model, dev_loader, "intact")
        stopper.step(dev_auc, model, epoch)
        if stopper.should_stop:
            break

    stopper.restore(model)
    best_epoch = stopper.best_epoch + 1

    # --- four-way ablation on the test fold ---
    auc_intact  = evaluate_in_mode(model, test_loader, "intact")
    auc_zero    = evaluate_in_mode(model, test_loader, "zero")
    auc_shuffle = evaluate_in_mode(model, test_loader, "shuffle")
    auc_cough   = evaluate_in_mode(model, test_loader, "cough")

    return {
        "test_set":      test_fold,
        "dev_set":       dev_fold,
        "best_epoch":    best_epoch,
        "auc_intact":    auc_intact,
        "auc_zero":      auc_zero,
        "auc_shuffle":   auc_shuffle,
        "auc_cough":     auc_cough,
        "delta_zero":    auc_intact - auc_zero,
        "delta_shuffle": auc_intact - auc_shuffle,
        "delta_cough":   auc_intact - auc_cough,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_ablation_plots(df, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    cols   = ["auc_intact", "auc_zero", "auc_shuffle", "auc_cough"]
    labels = ["Intact", "Zero", "Shuffle", "Cough"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # --- boxplot of the four AUCs ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bp = ax.boxplot(
        [df[c].dropna().values for c in cols],
        labels=labels,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=0.9),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="grey",
                        markeredgecolor="none"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, c in enumerate(cols, start=1):
        vals = df[c].dropna().values
        x = np.random.default_rng(42).normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, s=6, color="black", alpha=0.25, zorder=3)

    ax.set_ylabel("Test AUC")
    ax.set_title("Cross-modal ablation (late fusion)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    box_path = os.path.join(figures_dir, "ablation_boxplot.pdf")
    fig.savefig(box_path, bbox_inches="tight")
    fig.savefig(box_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {box_path}")
    print(f"Wrote {box_path.replace('.pdf', '.png')}")

    # --- bar chart of the three deltas ---
    delta_cols   = ["delta_zero", "delta_shuffle", "delta_cough"]
    delta_labels = ["Intact – Zero", "Intact – Shuffle", "Intact – Cough"]
    means = [df[c].mean() for c in delta_cols]
    stds  = [df[c].std()  for c in delta_cols]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(delta_cols))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#DD8452", "#55A868", "#C44E52"],
                  alpha=0.75, edgecolor="black", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(delta_labels)
    ax.set_ylabel(r"$\Delta$ AUC (intact minus ablated)")
    ax.set_title("Effect of removing the second branch")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m,
                f"{m:+.3f}",
                ha="center", va="bottom" if m >= 0 else "top",
                fontsize=9)

    fig.tight_layout()
    delta_path = os.path.join(figures_dir, "ablation_deltas.pdf")
    fig.savefig(delta_path, bbox_inches="tight")
    fig.savefig(delta_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {delta_path}")
    print(f"Wrote {delta_path.replace('.pdf', '.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N fold pairs.")
    parser.add_argument("--fold", type=int, nargs=2, default=None,
                        metavar=("TEST", "DEV"),
                        help="Run a single fold pair.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.fold is not None:
        pairs = [(args.fold[0], args.fold[1])]
    else:
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        if args.limit:
            pairs = pairs[:args.limit]

    print(f"Running ablation on {len(pairs)} fold pair(s)")
    print(f"Config: lr={CFG['lr']} wd={CFG['wd']} aug={CFG['augmentation']} "
          f"pretrained={CFG['use_pretrained']}")
    print()

    rows = []
    for n, (test_fold, dev_fold) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] ", end="", flush=True)
        try:
            row = train_and_ablate(test_fold, dev_fold)
            rows.append(row)
            print(f"intact={row['auc_intact']:.4f}  "
                  f"zero={row['auc_zero']:.4f}  "
                  f"shuffle={row['auc_shuffle']:.4f}  "
                  f"cough={row['auc_cough']:.4f}  "
                  f"(epoch {row['best_epoch']})")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    if not rows:
        print("No successful runs.")
        return

    df = pd.DataFrame(rows)
    per_fold_path = os.path.join(OUTPUT_DIR, "ablation_per_fold.csv")
    df.to_csv(per_fold_path, index=False)
    print(f"\nWrote {per_fold_path}")

    # --- summary ---
    cols = ["auc_intact", "auc_zero", "auc_shuffle", "auc_cough",
            "delta_zero", "delta_shuffle", "delta_cough"]
    summary = pd.DataFrame({
        "mean": df[cols].mean().round(4),
        "std":  df[cols].std().round(4),
    })
    summary_path = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
    summary.to_csv(summary_path)

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY  (mean ± std across fold pairs)")
    print("=" * 70)
    print(summary.to_string())
    print(f"\nFold pairs completed: {len(df)} / {len(pairs)}")

    # --- interpretation ---
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    d_zero    = df["delta_zero"].mean()
    d_shuffle = df["delta_shuffle"].mean()
    d_cough   = df["delta_cough"].mean()
    print(f"Intact - zero    = {d_zero:+.4f}   "
          f"(positive: model uses the second branch)")
    print(f"Intact - shuffle = {d_shuffle:+.4f}   "
          f"(positive: model uses real speech, not just the branch bias)")
    print(f"Intact - cough   = {d_cough:+.4f}   "
          f"(positive: speech adds beyond a second view of cough)")

    # --- plots ---
    try:
        make_ablation_plots(df, FIGURES_DIR)
    except Exception as exc:
        print(f"\nPlot generation failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()