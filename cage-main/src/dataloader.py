import os
import torch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ======================================================================
# ORIGINAL FUNCTIONS (unchanged)
# ======================================================================
def get_data(dataset, data_folds, i, j, cough_dir, loss, batch_size, num_outer_folds=10):
    """
    Returns train, val, test DataLoaders for single‑modality cough data.
    """
    train_set_files = [data_folds + f"/fold_{k}" for k in range(num_outer_folds) if k != j and k != i]
    dev_set_file = data_folds + f"/fold_{j}"
    test_set_file = data_folds + f"/fold_{i}"
    mean, std = get_mean_std(train_set_files, dataset, cough_dir, 128)

    train_data_set = ConcatDataset([
        CoughDatasetCleaned(dataset, file + ".csv", cough_dir, loss, mean, std, "none") for file in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_data = None
    if j is not None:
        val_data_set = CoughDatasetCleaned(dataset, dev_set_file + ".csv", cough_dir, loss, mean, std, "none")
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_data = None
    if i is not None:
        test_data_set = CoughDatasetCleaned(dataset, test_set_file + ".csv", cough_dir, loss, mean, std, "none")
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    return train_data, val_data, test_data


def get_mean_std(train_set_files, dataset, dir, inner_bins=128):
    """
    Compute mean/std for a modality (cough or speech).
    """
    train_data_set = EmptyDataset()
    for file in train_set_files:
        if dir.endswith("mel_spectrograms_128"):
            train_data_set = ConcatDataset([train_data_set, CoughDataset(dataset, file + ".csv", dir, inner_bins)])
        else:
            train_data_set = ConcatDataset([train_data_set, SpeechDataset(dataset, file + ".csv", dir, inner_bins)])
    mels = []
    train_loader = DataLoader(train_data_set, batch_size=100, num_workers=1)
    for images, _ in train_loader:
        mels.append(images)
    mels = torch.cat(mels, dim=0)
    perc = int(2/3 * mels.shape[1])
    mean = torch.mean(mels[:, :perc, :], dim=(0,1))
    std = torch.std(mels[:, :perc, :], dim=(0,1))
    return mean, std


class EmptyDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


class CoughDataset(Dataset):
    def __init__(self, dataset, annotations_file, dir, bins=128):
        self.labels = pd.read_csv(annotations_file)
        self.dir = dir
        self.dataset = dataset
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels["Status"][idx]
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image = torch.tensor(np.transpose(np.load(path)))
        image = image[:50, :]
        if self.dataset == "hyfe" and image.shape[0] < 40:
            image = torch.nn.functional.pad(image, (0,0,40-image.shape[0],0), "constant", 0)
        if self.dataset == "cage" and image.shape[0] < 50:
            image = torch.nn.functional.pad(image, (0,0,50-image.shape[0],0), "constant", 0)
        return image, label


class SpeechDataset(Dataset):
    """
    Loads speech spectrograms by scanning the patient folders.
    Uses the CSV only to get patient IDs.
    """
    def __init__(self, dataset, annotations_file, dir, bins=128):
        self.dir = dir
        self.dataset = dataset
        df = pd.read_csv(annotations_file)
        self.patient_ids = df["Cough_ID"].astype(str).apply(lambda x: x.split("/")[0]).unique().tolist()
        self.file_list = []
        for pid in self.patient_ids:
            folder = os.path.join(self.dir, pid)
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.endswith(".npy"):
                        self.file_list.append(os.path.join(folder, f))
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        path = self.file_list[idx]
        image = torch.tensor(np.transpose(np.load(path)))
        image = image[:50, :]
        if self.dataset == "hyfe" and image.shape[0] < 40:
            image = torch.nn.functional.pad(image, (0,0,40-image.shape[0],0), "constant", 0)
        if self.dataset == "cage" and image.shape[0] < 50:
            image = torch.nn.functional.pad(image, (0,0,50-image.shape[0],0), "constant", 0)
        return image, 0   # dummy label


class CoughDatasetCleaned(Dataset):
    def __init__(self, dataset, annotations_file, dir, loss, mean, std, fusion_type):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = dir
        self.loss = loss
        self.mean = mean
        self.std = std
        self.fusion_type = fusion_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.dataset == "hyfe" or self.dataset == "cage":
            label = self.labels["Status"][idx]
            path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
            image_raw = torch.tensor(np.transpose(np.load(path)))

            if self.loss == "cross_entropy":
                image = self.standardize(image_raw)
                image_mean = image.mean(0)
                return image_mean, label
            elif self.loss == 'cross_entropy_resnet':
                if self.fusion_type == "none":
                    return self.pad(self.standardize(self.repeat(image_raw))), label
                # For early fusion we need a different dataset (see below)
                raise NotImplementedError("Use EarlyFusionDataset for early fusion")

    def repeat(self, image):
        image = image[None, :, :]
        image = torch.concat((image, image, image))
        return image

    def pad(self, image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

    def standardize(self, image):
        image = (image - self.mean) / self.std
        return image


class SpeechDatasetCleaned(Dataset):
    def __init__(self, dataset, annotations_file, dir, loss, mean, std, fusion_type):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = dir
        self.loss = loss
        self.mean = mean
        self.std = std
        self.fusion_type = fusion_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.dataset == "hyfe" or self.dataset == "cage":
            label = self.labels["Status"][idx]
            image_list = []
            path = os.path.join(self.dir, str(self.labels["Speech_ID"][idx]) + ".npy")
            for images in np.load(path, allow_pickle=True):
                images = self.standardize(raw_images=torch.tensor(np.transpose(images)))
                images = self.pad(images)
                image_list.append(images)
            final_patient_image = torch.cat(image_list, dim=1)
            if self.fusion_type == "early":
                final_patient_image_mean = final_patient_image.mean(0).repeat(1, 50)
            elif self.fusion_type == "late":
                final_patient_image_mean = final_patient_image.mean(0)
            return final_patient_image_mean, label

    def standardize(self, raw_images):
        return (raw_images - self.mean) / self.std


class EarlyFusionFlatDataset(Dataset):
    """
    For each patient, compute the mean speech image (over all speech recordings).
    Then for each cough, create a 3‑channel fused image:
        ch1: cough (standardized, padded to 224x224)
        ch2: mean speech (standardized, padded to 224x224)
        ch3: ch1 * ch2
    Returns (fused_image, patient_label, patient_id) for each cough.
    """
    def __init__(self, annotations_file, cough_dir, speech_dir,
                 cough_mean, cough_std, speech_mean, speech_std, is_train=False):
        self.df = pd.read_csv(annotations_file)
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(lambda x: x.split('/')[0])
        self.df = self.df[self.df['Cough_ID'].astype(str).map(
            lambda cid: os.path.exists(os.path.join(cough_dir, cid + ".npy"))
        )].reset_index(drop=True)
        self.patients = self.df.groupby('patient_id').agg({'Cough_ID': list, 'Status': 'first'}).reset_index()
        # Keep only patients that have a speech folder with at least one file
        self.patients = self.patients[self.patients['patient_id'].map(
            lambda pid: os.path.isdir(os.path.join(speech_dir, pid)) and len(os.listdir(os.path.join(speech_dir, pid))) > 0
        )].reset_index(drop=True)
        # Flatten list of (cough_id, patient_id, label)
        self.samples = []
        for _, row in self.patients.iterrows():
            pid = row['patient_id']
            label = int(row['Status'])
            for cid in row['Cough_ID']:
                self.samples.append((cid, pid, label))
        self.cough_dir = cough_dir
        self.speech_dir = speech_dir
        self.cough_mean = cough_mean
        self.cough_std = cough_std
        self.speech_mean = speech_mean
        self.speech_std = speech_std
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def _pad_to_224(self, img):
        """
        Pad a (H,W) tensor to (224,224) using zero padding.
        """
        if img.ndim == 2:
            img = img.unsqueeze(0)   # (1,H,W)
        H, W = img.shape[-2], img.shape[-1]
        pad_h = max(0, 224 - H)
        pad_w = max(0, 224 - W)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        return img[0]  # (224,224)

    def _mean_speech_image(self, pid):
        folder = os.path.join(self.speech_dir, pid)
        imgs = []
        for fname in os.listdir(folder):
            if fname.endswith('.npy'):
                raw = torch.tensor(np.transpose(np.load(os.path.join(folder, fname))))
                norm = (raw - self.speech_mean) / self.speech_std
                imgs.append(self._pad_to_224(norm))
        if imgs:
            return torch.stack(imgs).mean(0)   # (224,224)
        else:
            return torch.zeros(224,224)

    def __getitem__(self, idx):
        cid, pid, label = self.samples[idx]
        # Load and process cough
        c_raw = torch.tensor(np.transpose(np.load(os.path.join(self.cough_dir, cid + ".npy"))))
        c_norm = (c_raw - self.cough_mean) / self.cough_std
        c_img = self._pad_to_224(c_norm)   # (224,224)

        # Mean speech for this patient
        m_speech = self._mean_speech_image(pid)   # (224,224)

        # Build 3 channels
        ch1 = c_img.unsqueeze(0)        # (1,224,224)
        ch2 = m_speech.unsqueeze(0)     # (1,224,224)
        ch3 = ch1 * ch2                 # product
        fused = torch.cat([ch1, ch2, ch3], dim=0)  # (3,224,224)

        # Gaussian noise augmentation for training only
        if self.is_train:
            fused = fused + torch.randn_like(fused) * 0.05   # std = 0.05

        return fused, label, pid   # now returns patient ID


def get_early_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size, num_outer_folds=10):
    """
    Returns DataLoaders for the early fusion flat dataset.
    Each sample is a (3,224,224) fused image with the patient's label and ID.
    """
    # Build two lists:
    # train_folds_noext: paths WITHOUT .csv (used by get_mean_std)
    # train_folds_csv: paths WITH .csv (used by dataset creation)
    train_folds_noext = [data_folds + f"/fold_{k}" for k in range(num_outer_folds) if k != j and k != i]
    train_folds_csv = [f + ".csv" for f in train_folds_noext]
    dev_file = data_folds + f"/fold_{j}.csv"
    test_file = data_folds + f"/fold_{i}.csv"

    # Compute means/stds on training folds (pass noext to avoid double .csv)
    cough_mean, cough_std = get_mean_std(train_folds_noext, dataset, cough_dir, 128)
    speech_mean, speech_std = get_speech_mean_std(train_folds_noext, dataset, speech_dir, 128)

    train_ds = ConcatDataset([EarlyFusionFlatDataset(f, cough_dir, speech_dir, cough_mean, cough_std, speech_mean, speech_std, is_train=True) for f in train_folds_csv])
    val_ds = EarlyFusionFlatDataset(dev_file, cough_dir, speech_dir, cough_mean, cough_std, speech_mean, speech_std, is_train=False) if j is not None else None
    test_ds = EarlyFusionFlatDataset(test_file, cough_dir, speech_dir, cough_mean, cough_std, speech_mean, speech_std, is_train=False) if i is not None else None

    def collate(batch):
        images, labels, pids = zip(*batch)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(pids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate) if test_ds else None
    return train_loader, val_loader, test_loader


def get_speech_mean_std(train_folds_noext, dataset, speech_dir, inner_bins=128):
    """
    Compute mean/std for speech by scanning all speech files of training patients.
    Expects `train_folds_noext` (list of fold paths WITHOUT .csv).
    """
    train_data_set = EmptyDataset()
    for fold in train_folds_noext:
        train_data_set = ConcatDataset([train_data_set, SpeechDataset(dataset, fold + ".csv", speech_dir, inner_bins)])
    mels = []
    loader = DataLoader(train_data_set, batch_size=100, num_workers=1)
    for images, _ in loader:
        mels.append(images)
    if not mels:
        raise ValueError("No speech files found for mean/std computation.")
    mels = torch.cat(mels, dim=0)
    perc = int(2/3 * mels.shape[1])
    mean = torch.mean(mels[:, :perc, :], dim=(0,1))
    std = torch.std(mels[:, :perc, :], dim=(0,1))
    return mean, std

def time_mask(image, T=30):
    """
    Apply time masking to a spectrogram image.
    Randomly selects a time segment of length up to T and masks it.
    """
    num_time_steps = image.shape[1]
    t = np.random.randint(0, T)
    t0 = np.random.randint(0, num_time_steps - t)
    image[:, t0:t0+t] = 0
    return image
def frequency_mask(image, F=13):
    """
    Apply frequency masking to a spectrogram image.
    Randomly selects a frequency segment of length up to F and masks it.
    """
    num_freq_bins = image.shape[0]
    f = np.random.randint(0, F)
    f0 = np.random.randint(0, num_freq_bins - f)
    image[f0:f0+f, :] = 0
    return image