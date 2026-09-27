import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF


# ======================================================================
# AUGMENTATION FUNCTIONS
# ======================================================================

def gaussian_noise(image, std=0.05):
    """Additive white noise with standard deviation `std`.

    Applied after the [0, 1] rescale, so `std = 0.05` is 5% of the signal
    range.
    """
    return image + torch.randn_like(image) * std


def solarisation(image, threshold=0.5):
    """Invert values above `threshold`. Returns a new tensor (no in-place).

    Applied after the [0, 1] rescale, so `threshold = 0.5` inverts the upper
    half of the rescaled range.
    """
    out  = image.clone()
    mask = out > threshold
    out[mask] = -out[mask]
    return out


def apply_augmentation(image, aug_type):
    if aug_type == "none":
        return image
    elif aug_type == "gaussian_noise":
        return gaussian_noise(image)
    elif aug_type == "solarisation":
        return solarisation(image, threshold=0.5)
    else:
        raise ValueError(f"Unknown augmentation: {aug_type}")


# ======================================================================
# SINGLE-MODALITY (COUGH-ONLY) DATASET
# ======================================================================

class CoughDatasetCleaned(Dataset):
    """
    Cough-only dataset used for both the single-modality ResNet-18 baseline
    and the single-modality logistic-regression baseline.

    Input pipeline:
        1. Load raw log-mel spectrogram  [128, 43]  (no per-bin standardisation)
        2. Pad to 224x224 (centre-crop if larger, zero-pad if smaller)
        3. Min-max rescale to [0, 1] using bounds from the padded image
        4. Augment (training only)
        5. Repeat to 3 channels
        6. ImageNet channel normalisation  -- only when `pretrained=True`
    """
    def __init__(self, dataset, annotations_file, dir, loss,
                 fusion_type, is_train=False, augmentation="none",
                 pretrained=False):
        self.labels = pd.read_csv(annotations_file)
        self.labels["patient_id"] = self.labels["Cough_ID"].astype(str).apply(
            lambda x: x.split("/")[0]
        )
        self.dataset      = dataset
        self.dir          = dir
        self.loss         = loss
        self.fusion_type  = fusion_type
        self.is_train     = is_train
        self.augmentation = augmentation
        self.pretrained   = pretrained

    def __len__(self):
        return len(self.labels)

    def _pad_to_224(self, img):
        """Centre-crop if larger than 224, then zero-pad to exactly 224x224."""
        if img.ndim == 2:
            img = img.unsqueeze(0)                 # [1, H, W]
        H, W = img.shape[-2], img.shape[-1]

        if H > 224:
            top = (H - 224) // 2
            img = img[:, top:top + 224, :]
            H   = 224
        if W > 224:
            left = (W - 224) // 2
            img  = img[:, :, left:left + 224]
            W    = 224

        pad_h = max(0, 224 - H)
        pad_w = max(0, 224 - W)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        return img[0]

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]
        pid   = self.labels["patient_id"][idx]
        path  = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")

        # .npy is stored as [freq, time] = [128, 43]; do NOT transpose.
        image_raw = torch.tensor(np.load(path), dtype=torch.float32)

        if self.loss == "cross_entropy":
            # Logistic-regression baseline: time-averaged raw log-mel -> [128]
            image = image_raw.mean(dim=1)
            return image, label, pid

        elif self.loss == "cross_entropy_resnet":
            image = self._pad_to_224(image_raw)                 # [224, 224]
            c_min, c_max = image.min(), image.max()
            image = (image - c_min) / (c_max - c_min + 1e-8)    # [0, 1]
            if self.is_train and self.augmentation != "none":
                image = apply_augmentation(image, self.augmentation)
            image = image.unsqueeze(0).repeat(3, 1, 1)          # [3, 224, 224]
            
            image = TF.normalize(image,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            return image, label, pid


def get_data(dataset, data_folds, i, j, cough_dir, loss, batch_size,
             num_outer_folds=10, augmentation="none", pretrained=False):
    train_set_files = [data_folds + f"/fold_{k}" for k in range(num_outer_folds)
                       if k != j and k != i]
    dev_set_file  = data_folds + f"/fold_{j}"
    test_set_file = data_folds + f"/fold_{i}"

    train_data_set = ConcatDataset([
        CoughDatasetCleaned(dataset, file + ".csv", cough_dir, loss,
                            "none", is_train=True, augmentation=augmentation,
                            pretrained=pretrained)
        for file in train_set_files
    ])
    val_ds = CoughDatasetCleaned(dataset, dev_set_file + ".csv", cough_dir, loss,
                                 "none", is_train=False, augmentation=augmentation,
                                 pretrained=pretrained) \
        if j is not None else None
    test_ds = CoughDatasetCleaned(dataset, test_set_file + ".csv", cough_dir, loss,
                                  "none", is_train=False, augmentation=augmentation,
                                  pretrained=pretrained) \
        if i is not None else None

    def collate(batch):
        images, labels, pids = zip(*batch)
        return (torch.stack(images),
                torch.tensor(labels, dtype=torch.long),
                list(pids))

    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4,
                            shuffle=True, drop_last=True, collate_fn=collate)
    val_data = DataLoader(val_ds, batch_size=batch_size, num_workers=4,
                          shuffle=False, collate_fn=collate) if val_ds else None
    test_data = DataLoader(test_ds, batch_size=batch_size, num_workers=4,
                           shuffle=False, collate_fn=collate) if test_ds else None

    return train_data, val_data, test_data


# ======================================================================
# EARLY-FUSION DATASET
# ======================================================================

class EarlyFusionFlatDataset(Dataset):
    """
    Early-fusion dataset. Produces a 3-channel input:
        channel 0: cough (scaled to [0, 1])
        channel 1: cough (duplicate)
        channel 2: patient-level mean speech (already in [0, 1])

    Speech is NOT augmented because it is a broadcast of a single spectral
    vector (rank-1 tensor); masking its time axis would be a no-op or create
    a spurious time-varying signal.

    ImageNet channel normalisation is applied ONLY when `pretrained=True`.
    """
    def __init__(self, annotations_file, cough_dir, speech_dir,
                 arch, is_train=False, augmentation="none",
                 pretrained=False):
        self.df = pd.read_csv(annotations_file)
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(
            lambda x: x.split('/')[0]
        )
        self.df = self.df[self.df['Cough_ID'].astype(str).map(
            lambda cid: os.path.exists(os.path.join(cough_dir, cid + ".npy"))
        )].reset_index(drop=True)

        self.patients = self.df.groupby('patient_id').agg(
            {'Cough_ID': list, 'Status': 'first'}
        ).reset_index()
        self.patients = self.patients[self.patients['patient_id'].map(
            lambda pid: os.path.exists(os.path.join(speech_dir, f"{pid}.pt"))
        )].reset_index(drop=True)

        self.arch    = arch
        self.samples = []
        for _, row in self.patients.iterrows():
            pid   = row['patient_id']
            label = int(row['Status'])
            for cid in row['Cough_ID']:
                self.samples.append((cid, pid, label))

        self.cough_dir    = cough_dir
        self.speech_dir   = speech_dir
        self.is_train     = is_train
        self.augmentation = augmentation
        self.pretrained   = pretrained

    def __len__(self):
        return len(self.samples)

    def _pad_to_224(self, img):
        if img.ndim == 2:
            img = img.unsqueeze(0)
        H, W = img.shape[-2], img.shape[-1]

        if H > 224:
            top = (H - 224) // 2
            img = img[:, top:top + 224, :]
            H   = 224
        if W > 224:
            left = (W - 224) // 2
            img  = img[:, :, left:left + 224]
            W    = 224

        pad_h = max(0, 224 - H)
        pad_w = max(0, 224 - W)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        return img[0]

    def _mean_speech_image(self, pid):
        pt_path = os.path.join(self.speech_dir, f"{pid}.pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, weights_only=True)
        raise ValueError(f"No preprocessed speech tensor for patient {pid}")

    def __getitem__(self, idx):
        cid, pid, label = self.samples[idx]

        # Cough: [128, 43] = [freq, time]  (no transpose)
        c_raw = torch.tensor(
            np.load(os.path.join(self.cough_dir, cid + ".npy")),
            dtype=torch.float32,
        )

        if self.arch == "resnet":
            # ----- Cough pipeline -----
            c_img = self._pad_to_224(c_raw)                       # [224, 224]
            c_min, c_max = c_img.min(), c_img.max()
            c_scaled = (c_img - c_min) / (c_max - c_min + 1e-8)   # [0, 1]
            if self.is_train and self.augmentation != "none":
                c_scaled = apply_augmentation(c_scaled, self.augmentation)

            # ----- Speech pipeline -----
            m_speech = self._mean_speech_image(pid)               # [128, 43]
            m_speech = self._pad_to_224(m_speech)                 # [224, 224]
            s_min, s_max = m_speech.min(), m_speech.max()
            m_speech = (m_speech - s_min) / (s_max - s_min + 1e-8)  # [0, 1]
            # ----- Fuse -----
            fused = torch.stack([c_scaled, c_scaled, m_speech], dim=0)
            if self.pretrained:
                fused = TF.normalize(fused,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        '''elif self.arch == "lr":
            # Logistic-regression variant: bilinear-resize cough to 224x224,
            # flatten both streams, concatenate.
            c_4d = c_raw.unsqueeze(0).unsqueeze(0)                # [1, 1, 128, 43]
            c_resized = torch.nn.functional.interpolate(
                c_4d, size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)                               # [224, 224]

            m_speech = self._mean_speech_image(pid)               # [128, 43]
            m_speech = self._pad_to_224(m_speech)                 # [224, 224]

            c_flat = torch.flatten(c_resized)                     # [50176]
            m_flat = torch.flatten(m_speech)                      # [50176]
            fused  = torch.cat([c_flat, m_flat], dim=0)   '''        # [100352]

        return fused, label, pid


def get_early_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss,
                          batch_size, arch, num_outer_folds=10, augmentation="none",
                          pretrained=False):
    train_folds_noext = [data_folds + f"/fold_{k}" for k in range(num_outer_folds)
                         if k != j and k != i]
    train_folds_csv = [f + ".csv" for f in train_folds_noext]
    dev_file  = data_folds + f"/fold_{j}.csv"
    test_file = data_folds + f"/fold_{i}.csv"

    train_ds = ConcatDataset([
        EarlyFusionFlatDataset(f, cough_dir, speech_dir, arch,
                               is_train=True, augmentation=augmentation,
                               pretrained=pretrained)
        for f in train_folds_csv
    ])
    val_ds = EarlyFusionFlatDataset(dev_file, cough_dir, speech_dir, arch,
                                    is_train=False, augmentation=augmentation,
                                    pretrained=pretrained) \
        if j is not None else None
    test_ds = EarlyFusionFlatDataset(test_file, cough_dir, speech_dir, arch,
                                     is_train=False, augmentation=augmentation,
                                     pretrained=pretrained) \
        if i is not None else None

    def collate(batch):
        images, labels, pids = zip(*batch)
        return (torch.stack(images),
                torch.tensor(labels, dtype=torch.long),
                list(pids))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=False, collate_fn=collate) \
        if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=False, collate_fn=collate) \
        if test_ds else None
    return train_loader, val_loader, test_loader