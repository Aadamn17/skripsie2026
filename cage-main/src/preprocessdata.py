import os
import torch
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm

RAW_SPEECH_DIR = "data/cage/mel_spectrograms_counting_128"
OUT_SPEECH_DIR = "data/cage/preprocessed_speech"
TARGET_SIZE = 128

os.makedirs(OUT_SPEECH_DIR, exist_ok=True)

def pad_to_224(img):
    if img.ndim == 2:
        img = img.unsqueeze(0)
    H, W = img.shape[-2], img.shape[-1]
    pad_h = max(0, TARGET_SIZE - H)
    pad_w = max(0, TARGET_SIZE - W)
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
    return img[0]

def preprocess_and_save():
    patient_dirs = [d for d in os.listdir(RAW_SPEECH_DIR) if os.path.isdir(os.path.join(RAW_SPEECH_DIR, d))]
    
    for pid in tqdm(patient_dirs, desc="Processing Patient Speech"):
        p_dir = os.path.join(RAW_SPEECH_DIR, pid)
        speech_files = [f for f in os.listdir(p_dir) if f.endswith(".npy")]
        
        if not speech_files:
            # Fallback to a zero-vector of length 128 (assuming 128 Mel bins)
            mean_patient_vector = torch.zeros(128)
        else:
            patient_vectors = []
            for f in speech_files:
                # Load raw array, e.g., shape [128, T]
                arr = np.load(os.path.join(p_dir, f))
                tensor = torch.tensor(arr, dtype=torch.float32) 
                
                # Rescale to [0, 1] before pooling
                t_min, t_max = tensor.min(), tensor.max()
                if t_max > t_min:
                    tensor = (tensor - t_min) / (t_max - t_min)
                
                # AVERAGE POOL OVER TIME (Axis 1) -> Shape becomes [128]
                time_averaged_vector = tensor.mean(dim=1) 
                patient_vectors.append(time_averaged_vector)
            
            # Average the 1D vectors across the patient's recordings -> Final shape [128]
            mean_patient_vector = torch.stack(patient_vectors).mean(0)
            mean_patient_vector = mean_patient_vector.unsqueeze(1).repeat(1,43)
            
        torch.save(mean_patient_vector, os.path.join(OUT_SPEECH_DIR, f"{pid}.pt"))

if __name__ == "__main__":
    preprocess_and_save()