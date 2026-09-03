#!/usr/bin/env python3
"""
Visualise raw spectrograms (128 bins) for a random patient, standardised to 50 time frames.

Usage:
    python visualize_patient.py                    # random patient from fold_0.csv
    python visualize_patient.py --patient_id 123
    python visualize_patient.py --csv data/cage/data_folds_filtered/fold_1.csv
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURATION – adjust these to match your dataset
# ----------------------------------------------------------------------
COUGH_DIR = "data/cage/mel_spectrograms_128"
SPEECH_DIR = "data/cage/mel_spectrograms_counting_128"
DEFAULT_CSV = "data/cage/data_folds_filtered/fold_0.csv"
TARGET_TIME_FRAMES = 43

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def get_all_patient_ids(csv_file):
    df = pd.read_csv(csv_file)
    patient_ids = df['Cough_ID'].astype(str).apply(lambda x: x.split('/')[0]).unique().tolist()
    return patient_ids

def find_cough_files(patient_id, cough_dir, csv_file):
    df = pd.read_csv(csv_file)
    patient_coughs = df[df['Cough_ID'].astype(str).str.startswith(f"{patient_id}/")]
    paths = []
    for _, row in patient_coughs.iterrows():
        cough_id = str(row['Cough_ID'])
        full_path = os.path.join(cough_dir, cough_id + ".npy")
        if os.path.exists(full_path):
            paths.append(full_path)
    return paths

def find_speech_files(patient_id, speech_dir):
    patient_folder = os.path.join(speech_dir, str(patient_id))
    if not os.path.isdir(patient_folder):
        return []
    files = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.npy')]
    return files

def load_spectrogram(path):
    """
    Load a .npy spectrogram.
    Saved as (freq_bins, time_frames) – no transpose needed.
    """
    data = np.load(path)
    return data   # shape (128, T)

def standardise_time_axis(spec, target_frames):
    """
    Pad or truncate the time axis (second dimension) to exactly target_frames.
    """
    current_frames = spec.shape[1]
    if current_frames < target_frames:
        pad_width = ((0, 0), (0, target_frames - current_frames))
        spec = np.pad(spec, pad_width=pad_width, constant_values=0)
    elif current_frames > target_frames:
        spec = spec[:, :target_frames]
    return spec

def plot_spectrograms(cough_spec, speech_spec, patient_id):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = ax1.imshow(cough_spec, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(f"Patient {patient_id} – Cough\nShape {cough_spec.shape}")
    ax1.set_xlabel(f"Time frames (standardised to {TARGET_TIME_FRAMES})")
    ax1.set_ylabel("Mel frequency bins (0–127)")
    plt.colorbar(im1, ax=ax1, label="Log magnitude")
    
    im2 = ax2.imshow(speech_spec, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title(f"Patient {patient_id} – Speech (Counting)\nShape {speech_spec.shape}")
    ax2.set_xlabel(f"Time frames (standardised to {TARGET_TIME_FRAMES})")
    ax2.set_ylabel("Mel frequency bins (0–127)")
    plt.colorbar(im2, ax=ax2, label="Log magnitude")
    
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global TARGET_TIME_FRAMES   # Must be first

    parser = argparse.ArgumentParser(description="Visualise raw cough/speech spectrograms (128 bins, standardised to 50 time frames) for a patient.")
    parser.add_argument('--patient_id', type=str, help="Patient ID")
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV)
    parser.add_argument('--cough_idx', type=int, default=0)
    parser.add_argument('--speech_idx', type=int, default=0)
    parser.add_argument('--time_frames', type=int, default=TARGET_TIME_FRAMES)
    args = parser.parse_args()

    TARGET_TIME_FRAMES = args.time_frames

    if args.patient_id is None:
        all_patients = get_all_patient_ids(args.csv)
        if not all_patients:
            print(f"Error: No patients found in {args.csv}")
            return
        args.patient_id = random.choice(all_patients)
        print(f"No patient_id provided. Using random patient: {args.patient_id}")

    cough_files = find_cough_files(args.patient_id, COUGH_DIR, args.csv)
    speech_files = find_speech_files(args.patient_id, SPEECH_DIR)

    if not cough_files:
        print(f"Error: No cough files found for patient {args.patient_id}")
        return
    if not speech_files:
        print(f"Error: No speech files found for patient {args.patient_id}")
        return

    cough_spec_raw = load_spectrogram(cough_files[args.cough_idx])
    speech_spec_raw = load_spectrogram(speech_files[args.speech_idx])

    cough_spec = standardise_time_axis(cough_spec_raw, TARGET_TIME_FRAMES)
    speech_spec = standardise_time_axis(speech_spec_raw, TARGET_TIME_FRAMES)

    print(f"Loaded cough: {os.path.basename(cough_files[args.cough_idx])} "
          f"original {cough_spec_raw.shape} → {cough_spec.shape}")
    print(f"Loaded speech: {os.path.basename(speech_files[args.speech_idx])} "
          f"original {speech_spec_raw.shape} → {speech_spec.shape}")

    plot_spectrograms(cough_spec, speech_spec, args.patient_id)

if __name__ == "__main__":
    main()