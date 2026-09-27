"""
Visualise cough and speech spectrograms from the CAGE-TB dataset.

Usage
-----
    # Single patient
    python3 view_spectrograms.py CAGE0048
    python3 view_spectrograms.py CAGE0048 --max 5

    # Every patient
    python3 view_spectrograms.py --all
    python3 view_spectrograms.py --all --max 3
    python3 view_spectrograms.py --all --outdir "raw spectrograms"

By default, figures are saved to:
    <outdir>/<patient_id>/raw.png
    <outdir>/<patient_id>/pipeline.png

The default outdir is "raw spectrograms", so running with no --outdir gives:
    raw spectrograms/CAGE0048/raw.png
    raw spectrograms/CAGE0048/pipeline.png
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
COUGH_DIR               = "data/cage/mel_spectrograms_128"
SPEECH_DIR              = "data/cage/mel_spectrograms_counting_128"
PREPROCESSED_SPEECH_DIR = "data/cage/preprocessed_speech"


# ----------------------------------------------------------------------
# Preprocessing helpers (kept identical to dataloader.py so the
# visualisation matches the training pipeline exactly)
# ----------------------------------------------------------------------
def pad_to_224(img):
    """Centre-crop if larger than 224, then zero-pad to exactly 224x224."""
    if img.ndim == 2:
        img = img.unsqueeze(0)
    H, W = img.shape[-2], img.shape[-1]

    if H > 224:
        top = (H - 224) // 2
        img = img[:, top:top + 224, :]
        H = 224
    if W > 224:
        left = (W - 224) // 2
        img = img[:, :, left:left + 224]
        W = 224

    pad_h = max(0, 224 - H)
    pad_w = max(0, 224 - W)
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
    return img[0]


def preprocess_cough(cough_path):
    """Return the four pipeline stages for one cough."""
    raw      = torch.tensor(np.load(cough_path), dtype=torch.float32)   # [128, 43]
    padded   = pad_to_224(raw)                                          # [224, 224]
    c_min, c_max = padded.min(), padded.max()
    scaled   = (padded - c_min) / (c_max - c_min + 1e-8)                # [0, 1]
    normalized = TF.normalize(
        scaled.unsqueeze(0).repeat(3, 1, 1),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return raw, padded, scaled, normalized


def load_preprocessed_speech(pid):
    path = os.path.join(PREPROCESSED_SPEECH_DIR, f"{pid}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True)   # [128, 43] in [0, 1]


# ----------------------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------------------
def plot_spectrogram(ax, arr, title, vmin=None, vmax=None):
    im = ax.imshow(arr, aspect="auto", origin="lower",
                   cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time frames", fontsize=8)
    ax.set_ylabel("frequency bins", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ----------------------------------------------------------------------
# Figure 1 — raw spectrograms
# ----------------------------------------------------------------------
def figure_raw(pid, max_show):
    cough_dir  = os.path.join(COUGH_DIR, pid)
    speech_dir = os.path.join(SPEECH_DIR, pid)

    coughs = sorted(Path(cough_dir).glob("*.npy"))  if os.path.isdir(cough_dir)  else []
    speech = sorted(Path(speech_dir).glob("*.npy")) if os.path.isdir(speech_dir) else []
    coughs = coughs[:max_show]
    speech = speech[:max_show]

    if not coughs and not speech:
        print(f"  No files found for patient {pid} in {COUGH_DIR} or {SPEECH_DIR}.")
        return None

    n = max(len(coughs), len(speech), 1)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.6 * n), squeeze=False)
    fig.suptitle(f"Raw spectrograms — patient {pid}", fontsize=12)

    for i in range(n):
        if i < len(coughs):
            arr = np.load(coughs[i])
            plot_spectrogram(axes[i, 0], arr,
                             f"cough  {coughs[i].stem}  {tuple(arr.shape)}")
        else:
            axes[i, 0].axis("off")

        if i < len(speech):
            arr = np.load(speech[i])
            plot_spectrogram(axes[i, 1], arr,
                             f"speech  {speech[i].stem}  {tuple(arr.shape)}")
        else:
            axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ----------------------------------------------------------------------
# Figure 2 — pipeline stages
# ----------------------------------------------------------------------
def figure_pipeline(pid):
    cough_dir = os.path.join(COUGH_DIR, pid)
    coughs = sorted(Path(cough_dir).glob("*.npy")) if os.path.isdir(cough_dir) else []
    if not coughs:
        print(f"  No cough files for {pid}; skipping pipeline figure.")
        return None

    raw, padded, scaled, normalized = preprocess_cough(coughs[0])
    speech_tensor = load_preprocessed_speech(pid)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"Pipeline stages — patient {pid}, cough {coughs[0].stem}",
                 fontsize=12)

    # Row 1 — cough only
    plot_spectrogram(axes[0, 0], raw.numpy(),
                     f"1. raw cough  {tuple(raw.shape)}")
    plot_spectrogram(axes[0, 1], padded.numpy(),
                     f"2. padded to 224x224  {tuple(padded.shape)}")
    plot_spectrogram(axes[0, 2], scaled.numpy(),
                     f"3. rescaled to [0, 1]")

    # Row 2 — after ImageNet norm, plus speech
    plot_spectrogram(axes[1, 0], normalized[0].numpy(),
                     f"4. cough after ImageNet normalisation")

    if speech_tensor is not None:
        plot_spectrogram(axes[1, 1], speech_tensor.numpy(),
                         f"5. preprocessed speech  {tuple(speech_tensor.shape)}")

        speech_padded = pad_to_224(speech_tensor)
        fused_speech  = TF.normalize(
            speech_padded.unsqueeze(0).repeat(3, 1, 1),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        plot_spectrogram(axes[1, 2], fused_speech[0].numpy(),
                         f"6. speech after ImageNet normalisation")
    else:
        for j in (1, 2):
            axes[1, j].axis("off")
            axes[1, j].text(0.5, 0.5,
                            "no preprocessed speech tensor",
                            ha="center", va="center",
                            transform=axes[1, j].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ----------------------------------------------------------------------
# Per-patient worker
# ----------------------------------------------------------------------
def process_patient(pid, max_show, outdir):
    """Generate and save both figures for one patient."""
    save_dir = os.path.join(outdir, pid)
    os.makedirs(save_dir, exist_ok=True)

    fig1 = figure_raw(pid, max_show)
    fig2 = figure_pipeline(pid)

    if fig1:
        path = os.path.join(save_dir, "raw.png")
        fig1.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig1)

    if fig2:
        path = os.path.join(save_dir, "pipeline.png")
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualise cough and speech spectrograms for CAGE-TB.",
    )
    parser.add_argument("patient_id", type=str, nargs="?", default=None,
                        help="Patient ID, e.g. CAGE0048. "
                             "Omit when using --all.")
    parser.add_argument("--all", action="store_true",
                        help="Process every patient folder found in "
                             f"{COUGH_DIR}.")
    parser.add_argument("--max", type=int, default=5,
                        help="Max number of coughs and speech files to show "
                             "per patient (default: 5)")
    parser.add_argument("--outdir", type=str, default="raw spectrograms",
                        help="Root output directory; a subdirectory named "
                             "after each patient ID is created inside it "
                             "(default: 'raw spectrograms')")
    args = parser.parse_args()

    if args.all:
        if not os.path.isdir(COUGH_DIR):
            print(f"Cough directory not found: {COUGH_DIR}")
            return

        patient_ids = sorted(
            d for d in os.listdir(COUGH_DIR)
            if os.path.isdir(os.path.join(COUGH_DIR, d))
        )

        if not patient_ids:
            print(f"No patient folders found in {COUGH_DIR}")
            return

        print(f"Processing {len(patient_ids)} patients...")
        for i, pid in enumerate(patient_ids, 1):
            print(f"[{i}/{len(patient_ids)}] {pid}")
            try:
                process_patient(pid, args.max, args.outdir)
            except Exception as e:
                print(f"  [ERROR] {pid}: {e}")
                continue

        print(f"\nDone. Figures saved under: {args.outdir}/")

    else:
        if args.patient_id is None:
            parser.error("Provide a patient_id or use --all.")
        print(f"Processing {args.patient_id}...")
        process_patient(args.patient_id, args.max, args.outdir)
        print(f"saved {os.path.join(args.outdir, args.patient_id, 'raw.png')}")
        print(f"saved {os.path.join(args.outdir, args.patient_id, 'pipeline.png')}")


if __name__ == "__main__":
    main()