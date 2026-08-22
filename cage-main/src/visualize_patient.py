"""
visualize_patient.py
--------------------
Loads a random patient from a chosen fold, displays their spectrograms,
and shows the pooled feature vector that the model uses.
Saves plots as PNG files instead of showing windows.
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works everywhere)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataloader import PatientEarlyImageDataset, get_mean_std_cough, get_mean_std_speech
from model_scripts import PatientEarlyImageClassifier, SelfAttentionPool

# ----------------- CONFIGURATION -----------------
DATASET = "cage"
FOLD = 0                      # Choose fold (0–9)
DATA_FOLDS = f"data/{DATASET}/data_folds_filtered"
COUGH_DIR = f"data/{DATASET}/mel_spectrograms_128"
SPEECH_DIR = f"data/{DATASET}/mel_spectrograms_counting_128"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------

# 1. Compute mean/std from the training folds
train_folds = [f"data/{DATASET}/data_folds_filtered/fold_{k}" for k in range(10) if k != FOLD]
c_mean, c_std = get_mean_std_cough(train_folds, DATASET, COUGH_DIR)
s_mean, s_std = get_mean_std_speech(train_folds, DATASET, SPEECH_DIR)

# 2. Load the patient dataset for the chosen fold
dataset = PatientEarlyImageDataset(
    annotations_file=f"{DATA_FOLDS}/fold_{FOLD}.csv",
    cough_dir=COUGH_DIR,
    speech_dir=SPEECH_DIR,
    cough_mean=c_mean,
    cough_std=c_std,
    speech_mean=s_mean,
    speech_std=s_std,
    is_train=False  # no augmentation
)

# 3. Pick a random patient
idx = random.randint(0, len(dataset) - 1)
fused_images, label = dataset[idx]
patient_id = dataset.patients.iloc[idx]['patient_id']
print(f"Random patient: {patient_id}, label: {label} (0=negative, 1=positive)")

# 4. Initialize the model and extract features
model = PatientEarlyImageClassifier(num_classes=2, freeze_backbone=True).to(DEVICE)
model.eval()

features = []
with torch.no_grad():
    for fused_img in fused_images:
        img = fused_img.to(DEVICE).float().unsqueeze(0)
        feat = model.pool(model.backbone(img)).flatten(1)
        features.append(feat.cpu().numpy().flatten())

# 5. Save the fused 3‑channel image
first_pair = fused_images[0]
c_img = first_pair[0].cpu().numpy()
s_img = first_pair[1].cpu().numpy()
avg_img = first_pair[2].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(c_img, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title("Cough spectrogram")
axes[0].set_xlabel("Time frames")
axes[0].set_ylabel("Frequency bins")
axes[1].imshow(s_img, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title("Speech spectrogram")
axes[1].set_xlabel("Time frames")
axes[1].set_ylabel("Frequency bins")
axes[2].imshow(avg_img, aspect='auto', origin='lower', cmap='viridis')
axes[2].set_title("Average (channel 2)")
axes[2].set_xlabel("Time frames")
axes[2].set_ylabel("Frequency bins")
plt.tight_layout()
plt.savefig(f"patient_{patient_id}_fused.png", dpi=150)
plt.close(fig)
print(f"Saved: patient_{patient_id}_fused.png")

# 6. Save the pooled feature vector
pooled_feat = np.mean(features, axis=0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(pooled_feat)), pooled_feat, width=1.0)
ax.set_title(f"Pooled feature vector (512‑dim) for {patient_id}")
ax.set_xlabel("Feature index")
ax.set_ylabel("Value")
plt.tight_layout()
plt.savefig(f"patient_{patient_id}_features.png", dpi=150)
plt.close(fig)
print(f"Saved: patient_{patient_id}_features.png")

# 7. Save raw spectrograms (if available)
patient_row = dataset.patients.iloc[idx]
cough_ids = patient_row['Cough_ID']
speech_ids = [f.replace('.npy', '') for f in os.listdir(os.path.join(SPEECH_DIR, patient_id)) if f.endswith('.npy')]

if len(cough_ids) > 1:
    random.shuffle(cough_ids)
if len(speech_ids) > 1:
    random.shuffle(speech_ids)

n_pairs = min(len(cough_ids), len(speech_ids))
if n_pairs > 0:
    c_path = os.path.join(COUGH_DIR, str(cough_ids[0]) + ".npy")
    s_path = os.path.join(SPEECH_DIR, patient_id, speech_ids[0] + ".npy")
    if os.path.exists(c_path) and os.path.exists(s_path):
        raw_c = np.load(c_path)
        raw_s = np.load(s_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(raw_c, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Raw cough {cough_ids[0]}")
        axes[0].set_xlabel("Time frames")
        axes[0].set_ylabel("Frequency bins")
        axes[1].imshow(raw_s, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f"Raw speech {speech_ids[0]}")
        axes[1].set_xlabel("Time frames")
        axes[1].set_ylabel("Frequency bins")
        plt.tight_layout()
        plt.savefig(f"patient_{patient_id}_raw.png", dpi=150)
        plt.close(fig)
        print(f"Saved: patient_{patient_id}_raw.png")

print("Visualization complete. Check the PNG files in the current directory.")