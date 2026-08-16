import os
import torch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Tuple

# ======================================================================
# 1. Helper functions: compute mean & std per frequency bin for normalisation
# ======================================================================

def get_mean_std_cough(train_set_files, dataset, cough_dir, inner_bins=128):
    """
    Compute per‑frequency‑bin mean and standard deviation from raw cough spectrograms.
    Uses only the first 2/3 of time frames (common practice to avoid silent trailing).
    """
    train_data_set = EmptyDataset()
    for file in train_set_files:
        train_data_set = ConcatDataset([train_data_set, CoughDataset(dataset, file+".csv", cough_dir, inner_bins)])
    mels = []
    train_loader = DataLoader(train_data_set, batch_size=100, num_workers=1)
    for images, _ in train_loader:
        mels.append(images)
    mels = torch.cat(mels, dim=0)
    perc = int(2/3 * mels.shape[1])           # use first 2/3 of time frames
    mean = torch.mean(mels[:,:perc,:], dim=(0,1))
    std = torch.std(mels[:,:perc,:], dim=(0,1))
    return mean, std

def get_mean_std_speech(train_set_files, dataset, speech_dir, inner_bins=128):
    """Same as above, but for speech spectrograms."""
    train_data_set = EmptyDataset()
    for file in train_set_files:
        train_data_set = ConcatDataset([train_data_set, SpeechDataset(dataset, file+".csv", speech_dir, inner_bins)])
    mels = []
    train_loader = DataLoader(train_data_set, batch_size=100, num_workers=1)
    for images, _ in train_loader:
        mels.append(images)
    mels = torch.cat(mels, dim=0)
    perc = int(2/3 * mels.shape[1])
    mean = torch.mean(mels[:,:perc,:], dim=(0,1))
    std = torch.std(mels[:,:perc,:], dim=(0,1))
    return mean, std

# ======================================================================
# 2. EmptyDataset – placeholder for concatenation
# ======================================================================

class EmptyDataset(Dataset):
    def __init__(self): pass
    def __len__(self): return 0
    def __getitem__(self, index): raise IndexError("Empty dataset cannot be indexed")

# ======================================================================
# 3. Raw dataset classes (no normalisation)
# ======================================================================

class CoughDataset(Dataset):
    """
    Loads raw cough spectrograms (as tensors) and labels.
    Used only to compute mean/std (not for training).
    """
    def __init__(self, dataset, annotations_file, cough_dir, bins=128):
        self.labels = pd.read_csv(annotations_file)
        self.dir = cough_dir
        self.dataset = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx] 
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image = torch.tensor(np.transpose(np.load(path)))
        image = image[:50, :]   # keep first 50 frequency bins (for compatibility)
        # Pad frequency dimension if needed (old code)
        if self.dataset == "hyfe" and image.shape[0] < 40:
            image = torch.nn.functional.pad(image, (0,0,(40-image.shape[0]),0), "constant", 0)
        if self.dataset == "cage" and image.shape[0] < 50:
            image = torch.nn.functional.pad(image, (0,0,(50-image.shape[0]),0), "constant", 0)
        return image, label

class SpeechDataset(Dataset):
    """Same as CoughDataset but for speech spectrograms."""
    def __init__(self, dataset, annotations_file, speech_dir, bins=128):
        self.labels = pd.read_csv(annotations_file)
        self.dir = speech_dir
        self.dataset = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx] 
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image = torch.tensor(np.transpose(np.load(path)))
        image = image[:50, :]
        if self.dataset == "hyfe" and image.shape[0] < 40:
            image = torch.nn.functional.pad(image, (0,0,(40-image.shape[0]),0), "constant", 0)
        if self.dataset == "cage" and image.shape[0] < 50:
            image = torch.nn.functional.pad(image, (0,0,(50-image.shape[0]),0), "constant", 0)
        return image, label

# ======================================================================
# 4. Cleaned dataset classes (standardised, padded, repeated for ResNet)
# ======================================================================

class CoughDatasetCleaned(Dataset):
    """
    For cough‑only models:
    - Standardises raw spectrogram using training mean/std.
    - If loss == 'cross_entropy' (LR): pools over time → (128,) vector.
    - If loss == 'cross_entropy_resnet' (ResNet): repeats to 3 channels,
      pads to 224×224 → (3,224,224) image.
    """
    def __init__(self, dataset, annotations_file, cough_dir, loss, mean, std):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = cough_dir
        self.loss = loss
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]   
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image_raw = torch.tensor(np.transpose(np.load(path)))
        # Standardise
        image = (image_raw - self.mean) / self.std
        if self.loss == "cross_entropy":          # Logistic Regression
            image_mean = image.mean(0)            # (128,)
            return image_mean, label
        elif self.loss == 'cross_entropy_resnet': # ResNet
            image = self.repeat(image)            # (3, freq, time)
            image = self.pad(image)               # (3, 224, 224)
            return image, label

    def repeat(self, image):
        image = image[None, :, :]                 # (1, freq, time)
        image = torch.concat((image, image, image), dim=0)  # (3, freq, time)
        return image

    def pad(self, image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

class SpeechDatasetCleaned(Dataset):
    """Same as CoughDatasetCleaned, but for speech."""
    def __init__(self, dataset, annotations_file, speech_dir, loss, mean, std):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = speech_dir
        self.loss = loss
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]   
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image_raw = torch.tensor(np.transpose(np.load(path)))
        image = (image_raw - self.mean) / self.std
        if self.loss == "cross_entropy":
            image_mean = image.mean(0)
            return image_mean, label
        elif self.loss == 'cross_entropy_resnet':
            image = self.repeat(image)
            image = self.pad(image)
            return image, label

    def repeat(self, image):
        image = image[None, :, :]
        image = torch.concat((image, image, image), dim=0)
        return image

    def pad(self, image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

# ======================================================================
# 5. Early fusion dataset (per utterance pair)
# ======================================================================

class EarlyFusionDataset(Dataset):
    """
    For each cough‑speech pair (same Cough_ID), loads both spectrograms,
    standardises each with its own training stats, and fuses them.
    Returns:
        - For loss='cross_entropy' (LR): (128‑dim vector) = average of pooled cough and pooled speech.
        - For loss='cross_entropy_resnet' (ResNet): (3,224,224) image where channels are:
            cough, speech, and their average.
    """
    def __init__(self, annotations_file, cough_dir, speech_dir, loss,
                 cough_mean, cough_std, speech_mean, speech_std):
        self.labels = pd.read_csv(annotations_file)
        self.cough_dir = cough_dir
        self.speech_dir = speech_dir
        self.loss = loss
        self.cough_mean = cough_mean
        self.cough_std = cough_std
        self.speech_mean = speech_mean
        self.speech_std = speech_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]
        cough_id = str(self.labels["Cough_ID"][idx])

        # Load and standardise cough spectrogram
        cough_path = os.path.join(self.cough_dir, cough_id + ".npy")
        cough_raw = torch.tensor(np.transpose(np.load(cough_path)))
        cough_norm = (cough_raw - self.cough_mean) / self.cough_std

        # Load and standardise speech spectrogram
        speech_path = os.path.join(self.speech_dir, cough_id + ".npy")
        speech_raw = torch.tensor(np.transpose(np.load(speech_path)))
        speech_norm = (speech_raw - self.speech_mean) / self.speech_std

        if self.loss == "cross_entropy":          # Logistic Regression
            # Pool over time, then average the two 128‑dim vectors
            cough_pooled = cough_norm.mean(0)     # (128,)
            speech_pooled = speech_norm.mean(0)   # (128,)
            fused = (cough_pooled + speech_pooled) / 2   # (128,)
            return fused, label

        elif self.loss == "cross_entropy_resnet": # ResNet
            # Convert each standardised spectrogram to a (224,224) image
            cough_img = self._to_image(cough_norm)
            speech_img = self._to_image(speech_norm)
            avg_img = (cough_img + speech_img) / 2
            # Stack into 3 channels: [cough, speech, average]
            stacked = torch.stack([cough_img, speech_img, avg_img], dim=0)  # (3,224,224)
            return stacked, label

    def _to_image(self, tensor):
        """Convert a (freq, time) tensor to a (224,224) single‑channel image."""
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)      # (1, freq, time)
        tensor = tensor.unsqueeze(0)          # (1, 1, freq, time)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode='bilinear', align_corners=False
        )
        return tensor[0, 0, :, :]             # (224, 224)

# ======================================================================
# 6. Fused dataset for intermediate fusion (patient‑level, uses pooled vectors)
# ======================================================================

class FusedCoughSpeechDataset(Dataset):
    """
    For each cough‑speech pair (same Cough_ID):
    - Standardises and pools each spectrogram over time → (128,) vector per modality.
    - Concatenates them → (256,) fused vector.
    Used for intermediate fusion (not patient‑level aggregation).
    """
    def __init__(self, dataset, annotations_file, cough_dir, speech_dir, loss, 
                 cough_mean, cough_std, speech_mean, speech_std):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.cough_dir = cough_dir
        self.speech_dir = speech_dir
        self.loss = loss
        self.cough_mean = cough_mean
        self.cough_std = cough_std
        self.speech_mean = speech_mean
        self.speech_std = speech_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]
        cough_id = str(self.labels["Cough_ID"][idx])

        # Cough
        cough_path = os.path.join(self.cough_dir, cough_id + ".npy")
        cough_raw = torch.tensor(np.transpose(np.load(cough_path)))
        cough_norm = (cough_raw - self.cough_mean) / self.cough_std
        cough_pooled = cough_norm.mean(0)          # (128,)

        # Speech
        speech_path = os.path.join(self.speech_dir, cough_id + ".npy")
        speech_raw = torch.tensor(np.transpose(np.load(speech_path)))
        speech_norm = (speech_raw - self.speech_mean) / self.speech_std
        speech_pooled = speech_norm.mean(0)        # (128,)

        # Concatenate
        fused = torch.cat([cough_pooled, speech_pooled])  # (256,)
        return fused, label

# ======================================================================
# 7. Functions to compute mean/std for fused dataset (pooled features)
# ======================================================================

def get_fused_mean_std(train_set_files, dataset, cough_dir, speech_dir, bins=128):
    """
    Computes mean/std for the *pooled* (time‑averaged) cough and speech features.
    These statistics are used to normalise the pooled vectors in FusedCoughSpeechDataset.
    """
    cough_feats, speech_feats = [], []

    for file in train_set_files:
        labels = pd.read_csv(file + ".csv")
        for _, row in labels.iterrows():
            cough_id = str(row["Cough_ID"])
            
            # Cough pooled vector
            cough_path = os.path.join(cough_dir, cough_id + ".npy")
            cough_img = torch.tensor(np.transpose(np.load(cough_path)))
            cough_pooled = cough_img.mean(0)            # time average
            cough_feats.append(cough_pooled)
            
            # Speech pooled vector
            speech_path = os.path.join(speech_dir, cough_id + ".npy")
            speech_img = torch.tensor(np.transpose(np.load(speech_path)))
            speech_pooled = speech_img.mean(0)
            speech_feats.append(speech_pooled)

    cough_feats = torch.stack(cough_feats)    # (N, 128)
    speech_feats = torch.stack(speech_feats)  # (N, 128)

    cough_mean = cough_feats.mean(0)
    cough_std  = cough_feats.std(0)
    speech_mean = speech_feats.mean(0)
    speech_std  = speech_feats.std(0)

    return cough_mean, cough_std, speech_mean, speech_std

# ======================================================================
# 8. Data loader functions – main interface for experiments
# ======================================================================

def get_cough_data(dataset, data_folds, i, j, cough_dir, loss, batch_size, num_outer_folds=10):
    """
    Returns dataloaders for cough‑only models.
    Uses data_folds_filtered (patient‑disjoint splits).
    """
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)

    mean, std = get_mean_std_cough(train_set_files, dataset, cough_dir, 128)

    train_data_set = ConcatDataset([
        CoughDatasetCleaned(dataset, f + ".csv", cough_dir, loss, mean, std)
        for f in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_data = None
    if j is not None:
        val_data_set = CoughDatasetCleaned(dataset, dev_set_file + ".csv", cough_dir, loss, mean, std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_data = None
    if i is not None:
        test_data_set = CoughDatasetCleaned(dataset, test_set_file + ".csv", cough_dir, loss, mean, std)
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    return train_data, val_data, test_data

def get_speech_data(dataset, data_folds, i, j, speech_dir, loss, batch_size, num_outer_folds=10):
    """Same as get_cough_data, but for speech‑only models."""
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)

    mean, std = get_mean_std_speech(train_set_files, dataset, speech_dir, 128)

    train_data_set = ConcatDataset([
        SpeechDatasetCleaned(dataset, f + ".csv", speech_dir, loss, mean, std)
        for f in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_data = None
    if j is not None:
        val_data_set = SpeechDatasetCleaned(dataset, dev_set_file + ".csv", speech_dir, loss, mean, std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_data = None
    if i is not None:
        test_data_set = SpeechDatasetCleaned(dataset, test_set_file + ".csv", speech_dir, loss, mean, std)
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    return train_data, val_data, test_data

def get_early_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size, num_outer_folds=10):
    """
    Returns dataloaders for early fusion (per utterance pair).
    For loss='cross_entropy' → 128‑dim fused vector (LR).
    For loss='cross_entropy_resnet' → 3×224×224 image (ResNet).
    """
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)

    cough_mean, cough_std = get_mean_std_cough(train_set_files, dataset, cough_dir, 128)
    speech_mean, speech_std = get_mean_std_speech(train_set_files, dataset, speech_dir, 128)

    train_data_set = ConcatDataset([
        EarlyFusionDataset(f + ".csv", cough_dir, speech_dir, loss,
                           cough_mean, cough_std, speech_mean, speech_std)
        for f in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size,
                            num_workers=4, shuffle=True, drop_last=True)

    val_data = None
    if j is not None:
        val_data_set = EarlyFusionDataset(dev_set_file + ".csv", cough_dir, speech_dir, loss,
                                          cough_mean, cough_std, speech_mean, speech_std)
        val_data = DataLoader(val_data_set, batch_size=batch_size,
                              num_workers=4, shuffle=True, drop_last=True)

    test_data = None
    if i is not None:
        test_data_set = EarlyFusionDataset(test_set_file + ".csv", cough_dir, speech_dir, loss,
                                           cough_mean, cough_std, speech_mean, speech_std)
        test_data = DataLoader(test_data_set, batch_size=batch_size,
                               num_workers=4, shuffle=True, drop_last=True)

    return train_data, val_data, test_data

def get_fused_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size, num_outer_folds=10):
    """
    Returns dataloaders for intermediate fusion (per utterance pair, 256‑dim vector).
    Used with FusedCoughSpeechDataset and Logistic_Regression(input_dim=256).
    """
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)

    cough_mean, cough_std, speech_mean, speech_std = get_fused_mean_std(
        train_set_files, dataset, cough_dir, speech_dir
    )

    train_data_set = ConcatDataset([
        FusedCoughSpeechDataset(dataset, f + ".csv", cough_dir, speech_dir, loss,
                                cough_mean, cough_std, speech_mean, speech_std)
        for f in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size, 
                            num_workers=4, shuffle=True, drop_last=True)

    val_data = None
    if j is not None:
        val_data_set = FusedCoughSpeechDataset(dataset, dev_set_file + ".csv", 
                                               cough_dir, speech_dir, loss,
                                               cough_mean, cough_std, speech_mean, speech_std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, 
                              num_workers=4, shuffle=True, drop_last=True)

    test_data = None
    if i is not None:
        test_data_set = FusedCoughSpeechDataset(dataset, test_set_file + ".csv",
                                                cough_dir, speech_dir, loss,
                                                cough_mean, cough_std, speech_mean, speech_std)
        test_data = DataLoader(test_data_set, batch_size=batch_size,
                               num_workers=4, shuffle=True, drop_last=True)

    return train_data, val_data, test_data