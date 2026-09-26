import os
import torch
import numpy as np
from tqdm import tqdm

RAW_SPEECH_DIR = "data/cage/mel_spectrograms_counting_128"
OUT_SPEECH_DIR = "data/cage/preprocessed_speech"
N_MELS         = 128
N_TIME_FRAMES  = 43     # matches the 0.5 s cough at 48 kHz, 2048 FFT, 512 hop

os.makedirs(OUT_SPEECH_DIR, exist_ok=True)


def preprocess_and_save():
    """
    For each patient, compute the patient-level mean counting-speech spectrum.

    Per recording:  time-average the log-mel spectrogram -> [128] vector.
    Across recordings: arithmetic mean -> [128] patient mean spectrum.
    Final step: rescale the patient mean to [0, 1] and broadcast across the
    time axis to a [128, 43] tensor, so it has the same shape as a cough
    spectrogram before dataloader padding.
    """
    patient_dirs = [d for d in os.listdir(RAW_SPEECH_DIR)
                    if os.path.isdir(os.path.join(RAW_SPEECH_DIR, d))]

    for pid in tqdm(patient_dirs, desc="Processing Patient Speech"):
        p_dir = os.path.join(RAW_SPEECH_DIR, pid)
        speech_files = [f for f in os.listdir(p_dir) if f.endswith(".npy")]

        if not speech_files:
            # No speech available for this patient: save a zero tensor of the
            # same shape as a normal patient tensor so downstream code does
            # not crash on shape mismatch.
            mean_patient_vector = torch.zeros(N_MELS, N_TIME_FRAMES)
        else:
            # 1. Per-recording time average in raw log-mel space.
            patient_vectors = []
            for f in speech_files:
                arr = np.load(os.path.join(p_dir, f))              # [128, T]
                tensor = torch.tensor(arr, dtype=torch.float32)
                time_averaged_vector = tensor.mean(dim=1)          # [128]
                patient_vectors.append(time_averaged_vector)

            # 2. Average across recordings -> patient mean spectrum.
            mean_spectrum = torch.stack(patient_vectors).mean(0)   # [128]

            # 3. Rescale the patient mean to [0, 1] (deterministic).
            vmin, vmax = mean_spectrum.min(), mean_spectrum.max()
            if vmax > vmin:
                mean_spectrum = (mean_spectrum - vmin) / (vmax - vmin)

            # 4. Broadcast across time.
            mean_patient_vector = mean_spectrum.unsqueeze(1).repeat(1, N_TIME_FRAMES)

        torch.save(mean_patient_vector, os.path.join(OUT_SPEECH_DIR, f"{pid}.pt"))


if __name__ == "__main__":
    preprocess_and_save()