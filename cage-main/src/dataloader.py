import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
import torchaudio
# ======================================================================
# AUGMENTATION FUNCTIONS
# ======================================================================

def time_mask(image, T=30):
    masking = torchaudio.transforms.TimeMasking(time_mask_param = T)
    image = masking(image)
    return image

def frequency_mask(image, F=15):
    masking = torchaudio.transforms.FrequencyMasking(freq_mask_param = F)
    image = masking(image)
    return image

def gaussian_noise(image, std=0.05):
    return image + torch.randn_like(image) * std

def solarisation(image, threshold=0.0):
    mask = image > threshold
    image[mask] = -image[mask]
    return image

def apply_augmentation(image, aug_type):
    if aug_type == "none":
        return image
    elif aug_type == "gaussian_noise":
        return gaussian_noise(image)
    elif aug_type == "frequency_masking":
        return frequency_mask(image, F=15)
    elif aug_type == "time_masking":
        return time_mask(image, T=30)
    elif aug_type == "solarisation":
        return solarisation(image, threshold=0.0)
    else:
        raise ValueError(f"Unknown augmentation: {aug_type}")

# ======================================================================
# UTILITIES & SINGLE-MODALITY
# ======================================================================

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
        image = torch.tensor(np.transpose(np.load(path)), dtype=torch.float32)
        if self.dataset == "hyfe" and image.shape[0] < 40:
            image = torch.nn.functional.pad(image, (0,0,40-image.shape[0],0), "constant", 0)
        if self.dataset == "cage" and image.shape[0] < 50:
            image = torch.nn.functional.pad(image, (0,0,50-image.shape[0],0), "constant", 0)
        return image, label

def get_mean_std(train_set_files, dataset, dir, inner_bins=128):
    train_data_set = EmptyDataset()
    for file in train_set_files:
        train_data_set = ConcatDataset([train_data_set, CoughDataset(dataset, file + ".csv", dir, inner_bins)])
    
    loader = DataLoader(train_data_set, batch_size=32, num_workers=1)
    total_sum, total_sq_sum, total_count = 0.0, 0.0, 0

    for images, _ in loader:
        perc = int(2/3 * images.shape[1])
        sub_img = images[:, :perc, :]
        total_sum += sub_img.sum(dim=(0, 1))
        total_sq_sum += (sub_img ** 2).sum(dim=(0, 1))
        total_count += sub_img.shape[0] * sub_img.shape[1]

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean, std

class CoughDatasetCleaned(Dataset):
    def __init__(self, dataset, annotations_file, dir, loss, mean, std, fusion_type, augmentation="none"):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = dir
        self.loss = loss
        self.mean = mean
        self.std = std
        self.fusion_type = fusion_type
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image_raw = torch.tensor(np.transpose(np.load(path)), dtype=torch.float32)

        if self.loss == "cross_entropy":
            image = (image_raw - self.mean) / self.std
            return image.mean(0), label
        elif self.loss == 'cross_entropy_resnet':
            if self.fusion_type == "none":
                image = (image_raw - self.mean) / self.std
                image = image[None, :, :].repeat(3, 1, 1)
                if (image.shape[-1] < 224) or (image.shape[-2] < 224):
                    pad_h = max(0, 224 - image.shape[-2])
                    pad_w = max(0, 224 - image.shape[-1])
                    image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), "constant", 0)
                if self.augmentation != "none":
                    image = apply_augmentation(image, self.augmentation)
                return image, label

def get_data(dataset, data_folds, i, j, cough_dir, loss, batch_size, num_outer_folds=10, augmentation="none"):
    train_set_files = [data_folds + f"/fold_{k}" for k in range(num_outer_folds) if k != j and k != i]
    dev_set_file = data_folds + f"/fold_{j}"
    test_set_file = data_folds + f"/fold_{i}"
    mean, std = get_mean_std(train_set_files, dataset, cough_dir, 128)

    train_data_set = ConcatDataset([
        CoughDatasetCleaned(dataset, file + ".csv", cough_dir, loss, mean, std, "none", augmentation)
        for file in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)

    val_data = DataLoader(CoughDatasetCleaned(dataset, dev_set_file + ".csv", cough_dir, loss, mean, std, "none", augmentation),
                          batch_size=batch_size, num_workers=2, shuffle=False) if j is not None else None
    test_data = DataLoader(CoughDatasetCleaned(dataset, test_set_file + ".csv", cough_dir, loss, mean, std, "none", augmentation),
                           batch_size=batch_size, num_workers=2, shuffle=False) if i is not None else None

    return train_data, val_data, test_data

# ======================================================================
# FUSION DATASETS
# ======================================================================

class EarlyFusionFlatDataset(Dataset):
    def __init__(self, annotations_file, cough_dir, speech_dir, cough_mean, cough_std,arch, is_train=False, augmentation="none"):
        self.df = pd.read_csv(annotations_file)
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(lambda x: x.split('/')[0])
        self.df = self.df[self.df['Cough_ID'].astype(str).map(
            lambda cid: os.path.exists(os.path.join(cough_dir, cid + ".npy"))
        )].reset_index(drop=True)
        
        self.patients = self.df.groupby('patient_id').agg({'Cough_ID': list, 'Status': 'first'}).reset_index()
        self.patients = self.patients[self.patients['patient_id'].map(
            lambda pid: os.path.exists(os.path.join(speech_dir, f"{pid}.pt"))
        )].reset_index(drop=True)
        self.arch = arch
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
        self.is_train = is_train
        self.augmentation = augmentation

    def __len__(self):
        return len(self.samples)

    def _pad_to_224(self, img):
        if img.ndim == 2:
            img = img.unsqueeze(0)
        H, W = img.shape[-2], img.shape[-1]
        pad_h = max(0, 224 - H)
        pad_w = max(0, 224 - W)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        return img[0]

    def _mean_speech_image(self, pid):
        pt_path = os.path.join(self.speech_dir, f"{pid}.pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, weights_only=True)
        return torch.zeros((224, 224))

    def __getitem__(self, idx):
        cid, pid, label = self.samples[idx]

        c_raw = torch.tensor(np.transpose(np.load(os.path.join(self.cough_dir, cid + ".npy"))), dtype=torch.float32)
        c_norm = (c_raw - self.cough_mean) / self.cough_std
        if self.arch == "resnet":
            c_augmented = apply_augmentation(c_norm, self.augmentation) if self.is_train and self.augmentation != "none" else c_norm
            c_img = self._pad_to_224(c_augmented)

            c_min, c_max = c_img.min(), c_img.max()
            c_scaled = (c_img - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else c_img

            m_speech = self._mean_speech_image(pid)
            if self.is_train  and self.augmentation !="none":
                m_speech = apply_augmentation(m_speech,self.augmentation)

            ch1 = c_scaled.unsqueeze(0)
            ch2 = m_speech.unsqueeze(0)
            ch3 = torch.abs(ch1-ch2)
            fused = torch.cat([ch1, ch1, ch2], dim=0)
            fused = TF.normalize(fused, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.arch == "lr":
            # Resize cough spectrogram (H x W) to fixed 224x224 to match speech
            # c_norm is (H, W) with variable W (time). Use bilinear interpolation.
            c_4d = c_norm.unsqueeze(0).unsqueeze(0)               # (1, 1, H, W)
            c_resized = torch.nn.functional.interpolate(
                c_4d, size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)                               # (224, 224)

            m_speech = self._mean_speech_image(pid).float()
            if m_speech.ndim == 3:                                # (C, 224, 224) safety
                m_speech = m_speech.mean(0)

            c_flat = torch.flatten(c_resized)                     # (224*224,)
            m_flat = torch.flatten(m_speech)                      # (224*224,)

            fused = torch.cat([c_flat, m_flat], dim=0)            # (2*224*224,) = (100352,)
            
        return fused, label, pid


class LateFusionDataset(Dataset):
    def __init__(self, annotations_file, cough_dir, speech_dir, cough_mean, cough_std, speech_arch, is_train=False, augmentation="none"):
        self.df = pd.read_csv(annotations_file)
        
        # 1. Parse patient ID
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(lambda x: str(x).split('/')[0])
        
        # 2. Extract relative cough file path safely (handle missing or double .npy)
        def resolve_cough_path(cid):
            cid_str = str(cid)
            if not cid_str.endswith(".npy"):
                cid_str += ".npy"
            return os.path.join(cough_dir, cid_str)

        self.df['cough_exists'] = self.df['Cough_ID'].apply(lambda cid: os.path.exists(resolve_cough_path(cid)))
        self.df = self.df[self.df['cough_exists']].reset_index(drop=True)

        # 3. Check for matching speech .pt tensors
        self.df['speech_exists'] = self.df['patient_id'].apply(
            lambda pid: os.path.exists(os.path.join(speech_dir, f"{pid}.pt"))
        )
        self.df = self.df[self.df['speech_exists']].reset_index(drop=True)

        self.samples = []
        for _, row in self.df.iterrows():
            cid = str(row['Cough_ID'])
            if not cid.endswith(".npy"):
                cid += ".npy"
            pid = str(row['patient_id'])
            label = int(row['Status'])
            self.samples.append((cid, pid, label))

        # Diagnostic check if dataset is empty
        if len(self.samples) == 0:
            print(f"\n[WARNING] 0 samples loaded for fold file: {annotations_file}")
            print(f"Check if files exist in:\n - Cough Dir: {cough_dir}\n - Speech Dir: {speech_dir}\n")

        self.cough_dir = cough_dir
        self.speech_dir = speech_dir
        self.cough_mean = cough_mean
        self.cough_std = cough_std
        self.speech_arch = speech_arch
        self.is_train = is_train
        self.augmentation = augmentation

    def __len__(self):
        return len(self.samples)

    def _pad_to_224(self, img):
        if img.ndim == 2:
            img = img.unsqueeze(0)
        H, W = img.shape[-2], img.shape[-1]
        pad_h = max(0, 224 - H)
        pad_w = max(0, 224 - W)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        return img[0]

    def _mean_speech_image(self, pid):
        pt_path = os.path.join(self.speech_dir, f"{pid}.pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, weights_only=True)
        return torch.zeros((224, 224))

    def __getitem__(self, idx):
        cid, pid, label = self.samples[idx]

        cough_path = os.path.join(self.cough_dir, cid)
        c_raw = torch.tensor(np.transpose(np.load(cough_path)), dtype=torch.float32)
        c_norm = (c_raw - self.cough_mean) / self.cough_std
        c_augmented = apply_augmentation(c_norm, self.augmentation) if self.is_train and self.augmentation != "none" else c_norm
        c_img = self._pad_to_224(c_augmented)

        c_min, c_max = c_img.min(), c_img.max()
        c_scaled = (c_img - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else c_img
        
        stream1 = c_scaled.unsqueeze(0).repeat(3, 1, 1)
        stream1 = TF.normalize(stream1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        s_img = self._mean_speech_image(pid)

        if self.speech_arch == "resnet":
            stream2 = s_img.unsqueeze(0).repeat(3, 1, 1)
            stream2 = TF.normalize(stream2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.speech_arch == "lr":
            stream2 = s_img.unsqueeze(0)

        return stream1, stream2, label, pid

# ======================================================================
# DATA LOADER FUNCTIONS FOR FUSION
# ======================================================================

def get_early_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size, arch,  num_outer_folds=10, augmentation="none"):
    train_folds_noext = [data_folds + f"/fold_{k}" for k in range(num_outer_folds) if k != j and k != i]
    train_folds_csv = [f + ".csv" for f in train_folds_noext]
    dev_file = data_folds + f"/fold_{j}.csv"
    test_file = data_folds + f"/fold_{i}.csv"

    cough_mean, cough_std = get_mean_std(train_folds_noext, dataset, cough_dir, 128)

    train_ds = ConcatDataset([
        EarlyFusionFlatDataset(f, cough_dir, speech_dir, cough_mean, cough_std, arch, is_train=True, augmentation=augmentation)
        for f in train_folds_csv
    ])
    val_ds = EarlyFusionFlatDataset(dev_file, cough_dir, speech_dir, cough_mean, cough_std,arch, is_train=False, augmentation=augmentation) if j is not None else None
    test_ds = EarlyFusionFlatDataset(test_file, cough_dir, speech_dir, cough_mean, cough_std,arch, is_train=False, augmentation=augmentation) if i is not None else None

    def collate(batch):
        images, labels, pids = zip(*batch)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(pids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate) if test_ds else None
    return train_loader, val_loader, test_loader


def get_late_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size, num_outer_folds=10, augmentation="none", speech_arch="none"):
    train_folds_noext = [data_folds + f"/fold_{k}" for k in range(num_outer_folds) if k != j and k != i]
    train_folds_csv = [f + ".csv" for f in train_folds_noext]
    dev_file = data_folds + f"/fold_{j}.csv"
    test_file = data_folds + f"/fold_{i}.csv"

    cough_mean, cough_std = get_mean_std(train_folds_noext, dataset, cough_dir, 128)

    train_ds = ConcatDataset([
        LateFusionDataset(f, cough_dir, speech_dir, cough_mean, cough_std, speech_arch=speech_arch, is_train=True, augmentation=augmentation)
        for f in train_folds_csv
    ])
    val_ds = LateFusionDataset(dev_file, cough_dir, speech_dir, cough_mean, cough_std, speech_arch=speech_arch, is_train=False, augmentation=augmentation) if j is not None else None
    test_ds = LateFusionDataset(test_file, cough_dir, speech_dir, cough_mean, cough_std, speech_arch=speech_arch, is_train=False, augmentation=augmentation) if i is not None else None

    def collate(batch):
        stream1, stream2, labels, pids = zip(*batch)
        return torch.stack(stream1), torch.stack(stream2), torch.tensor(labels, dtype=torch.long), list(pids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate) if test_ds else None
    return train_loader, val_loader, test_loader
