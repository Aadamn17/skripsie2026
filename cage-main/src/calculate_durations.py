import os
import numpy as np
import pandas as pd
from scipy.io import wavfile

# ------------------------------------------------------------
# 1. Load labels
# ------------------------------------------------------------
labels_df = pd.read_csv("data/cage/labels.csv")
patient_label = dict(zip(labels_df["Patient_ID"], labels_df["TB_status"]))

# ------------------------------------------------------------
# 2. Helper: get patients with at least one .wav file
# ------------------------------------------------------------
def get_patients_with_files(directory):
    patients = set()
    for patient in os.listdir(directory):
        patient_path = os.path.join(directory, patient)
        if os.path.isdir(patient_path):
            if any(f.endswith('.wav') for f in os.listdir(patient_path)):
                patients.add(patient)
    return patients

# ------------------------------------------------------------
# 3. Get patient sets
# ------------------------------------------------------------
cough_dir = "data/cage/audio"
speech_dir = "data/cage/counting"

cough_patients = get_patients_with_files(cough_dir)
speech_patients = get_patients_with_files(speech_dir)
both_patients = cough_patients & speech_patients

# Separate by class
pos_patients = [p for p in both_patients if patient_label.get(p) == 1]
neg_patients = [p for p in both_patients if patient_label.get(p) == 0]

# ------------------------------------------------------------
# 4. Print patient counts
# ------------------------------------------------------------
print("=== Patients with both cough AND speech ===")
print(f"Total: {len(both_patients)}")
print(f"  Positive (TB+): {len(pos_patients)}")
print(f"  Negative (TB-): {len(neg_patients)}")

# ------------------------------------------------------------
# 5. Collect file paths for both‑modality patients
# ------------------------------------------------------------
def collect_file_paths(patient_ids, modality_dir):
    file_paths = []
    for pid in patient_ids:
        pdir = os.path.join(modality_dir, pid)
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            if fname.endswith('.wav'):
                file_paths.append(os.path.join(pdir, fname))
    return file_paths

cough_files = collect_file_paths(both_patients, cough_dir)
speech_files = collect_file_paths(both_patients, speech_dir)

print(f"\n--- Total segments ---")
print(f"  Cough files:   {len(cough_files)}")
print(f"  Speech files:  {len(speech_files)}")

# ------------------------------------------------------------
# 6. Get durations and labels for each file
# ------------------------------------------------------------
def get_durations_and_labels(file_list, patient_label_dict):
    durations = []
    labels = []
    for fpath in file_list:
        pid = os.path.basename(os.path.dirname(fpath))
        label = patient_label_dict.get(pid)
        if label is None:
            continue
        try:
            sr, data = wavfile.read(fpath)
            duration = len(data) / sr
            durations.append(duration)
            labels.append(label)
        except Exception as e:
            print(f"Warning: Could not read {fpath}: {e}")
    return np.array(durations), np.array(labels)

cough_durations, cough_labels = get_durations_and_labels(cough_files, patient_label)
speech_durations, speech_labels = get_durations_and_labels(speech_files, patient_label)

# ------------------------------------------------------------
# 7. Segment counts by class
# ------------------------------------------------------------
cough_pos_count = np.sum(cough_labels == 1)
cough_neg_count = np.sum(cough_labels == 0)
speech_pos_count = np.sum(speech_labels == 1)
speech_neg_count = np.sum(speech_labels == 0)

print("\n--- Cough segments ---")
print(f"  Positive (TB+): {cough_pos_count}")
print(f"  Negative (TB-): {cough_neg_count}")

print("\n--- Speech segments ---")
print(f"  Positive (TB+): {speech_pos_count}")
print(f"  Negative (TB-): {speech_neg_count}")

# ------------------------------------------------------------
# 8. Duration statistics helper
# ------------------------------------------------------------
def print_duration_stats(name, durations):
    if len(durations) == 0:
        print(f"{name}: No data")
        return
    print(f"\n--- {name} duration statistics (seconds) ---")
    print(f"  Mean:   {np.mean(durations):.3f}")
    print(f"  Median: {np.median(durations):.3f}")
    print(f"  Std:    {np.std(durations):.3f}")
    print(f"  Min:    {np.min(durations):.3f}")
    print(f"  Max:    {np.max(durations):.3f}")

# ------------------------------------------------------------
# 9. Duration stats split by class
# ------------------------------------------------------------
print_duration_stats("Cough (All)", cough_durations)
print_duration_stats("Cough (TB+)", cough_durations[cough_labels == 1])
print_duration_stats("Cough (TB-)", cough_durations[cough_labels == 0])

print_duration_stats("Speech (All)", speech_durations)
print_duration_stats("Speech (TB+)", speech_durations[speech_labels == 1])
print_duration_stats("Speech (TB-)", speech_durations[speech_labels == 0])

# ------------------------------------------------------------
# 10. Per‑patient averages split by class
# ------------------------------------------------------------
def count_files_per_patient(patient_ids, modality_dir):
    counts = {}
    for pid in patient_ids:
        pdir = os.path.join(modality_dir, pid)
        if os.path.isdir(pdir):
            counts[pid] = len([f for f in os.listdir(pdir) if f.endswith('.wav')])
        else:
            counts[pid] = 0
    return counts

cough_counts_all = count_files_per_patient(both_patients, cough_dir)
speech_counts_all = count_files_per_patient(both_patients, speech_dir)

cough_counts_pos = count_files_per_patient(pos_patients, cough_dir)
speech_counts_pos = count_files_per_patient(pos_patients, speech_dir)

cough_counts_neg = count_files_per_patient(neg_patients, cough_dir)
speech_counts_neg = count_files_per_patient(neg_patients, speech_dir)

print("\n--- Average segments per patient ---")
print(f"  All patients: Cough: {np.mean(list(cough_counts_all.values())):.2f}, Speech: {np.mean(list(speech_counts_all.values())):.2f}")
print(f"  TB+ patients: Cough: {np.mean(list(cough_counts_pos.values())):.2f}, Speech: {np.mean(list(speech_counts_pos.values())):.2f}")
print(f"  TB- patients: Cough: {np.mean(list(cough_counts_neg.values())):.2f}, Speech: {np.mean(list(speech_counts_neg.values())):.2f}")