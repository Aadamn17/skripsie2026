import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import pad_to_224, seed_worker, create_generator, standard_collate, late_fusion_collate


def per_frequency_normalize(spectrogram, eps=1e-8):
    mean = spectrogram.mean(dim=-1, keepdim=True)
    std  = spectrogram.std(dim=-1, keepdim=True)
    return (spectrogram - mean) / (std + eps)


def gaussian_noise(image, std=0.05):
    return image + torch.randn_like(image) * std


def solarisation(image, threshold=0.5):
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
    raise ValueError(f"Unknown augmentation: {aug_type}")


class CoughDatasetCleaned(Dataset):
    def __init__(self, dataset, annotations_file, dir, loss,
                 is_train=False, augmentation="none"):
        self.labels = pd.read_csv(annotations_file)
        self.labels["patient_id"] = self.labels["Cough_ID"].astype(str).apply(
            lambda x: x.split("/")[0]
        )
        self.dataset      = dataset
        self.dir          = dir
        self.loss         = loss
        self.is_train     = is_train
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels["Status"][idx]
        pid   = self.labels["patient_id"][idx]
        path  = os.path.join(self.dir, str(self.labels["Cough_ID"][idx]) + ".npy")
        image_raw = torch.tensor(np.load(path), dtype=torch.float32)

        if self.loss == "cross_entropy":
            #image_raw = per_frequency_normalize(image_raw)
            image = image_raw.mean(dim=1)
            return image, label, pid
        elif self.loss == "cross_entropy_resnet":
            image_raw = per_frequency_normalize(image_raw)
            image = pad_to_224(image_raw)
            if self.is_train and self.augmentation != "none":
                image = apply_augmentation(image, self.augmentation)
            image = image.unsqueeze(0).repeat(3, 1, 1)
            return image, label, pid


def get_data(dataset, data_folds, i, j, cough_dir, loss, batch_size,
             num_outer_folds=10, augmentation="none"):
    train_set_files = [data_folds + f"/fold_{k}"
                       for k in range(num_outer_folds) if k != j and k != i]

    train_data_set = ConcatDataset([
        CoughDatasetCleaned(dataset, f + ".csv", cough_dir, loss,
                            is_train=True, augmentation=augmentation)
        for f in train_set_files
    ])
    val_ds  = CoughDatasetCleaned(dataset, data_folds + f"/fold_{j}.csv",
                                  cough_dir, loss, is_train=False,
                                  augmentation=augmentation) if j is not None else None
    test_ds = CoughDatasetCleaned(dataset, data_folds + f"/fold_{i}.csv",
                                  cough_dir, loss, is_train=False,
                                  augmentation=augmentation) if i is not None else None

    g = create_generator()
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4,
                            pin_memory=True, persistent_workers=True,
                            shuffle=True, drop_last=True,
                            collate_fn=standard_collate,
                            worker_init_fn=seed_worker, generator=g)
    val_data  = DataLoader(val_ds, batch_size=batch_size, num_workers=4,
                           pin_memory=True, persistent_workers=True,
                           shuffle=False, collate_fn=standard_collate) if val_ds else None
    test_data = DataLoader(test_ds, batch_size=batch_size, num_workers=4,
                           pin_memory=True, persistent_workers=True,
                           shuffle=False, collate_fn=standard_collate) if test_ds else None
    return train_data, val_data, test_data


class EarlyFusionFlatDataset(Dataset):
    def __init__(self, annotations_file, cough_dir, speech_dir, arch,
                 is_train=False, augmentation="none"):
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

        self.arch, self.samples = arch, []
        for _, row in self.patients.iterrows():
            for cid in row['Cough_ID']:
                self.samples.append((cid, row['patient_id'], int(row['Status'])))

        self.cough_dir, self.speech_dir = cough_dir, speech_dir
        self.is_train, self.augmentation = is_train, augmentation

    def __len__(self):
        return len(self.samples)

    def _mean_speech_image(self, pid):
        pt_path = os.path.join(self.speech_dir, f"{pid}.pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, weights_only=True)
        raise ValueError(f"No preprocessed speech tensor for patient {pid}")

    def __getitem__(self, idx):
        cid, pid, label = self.samples[idx]
        c_raw = torch.tensor(np.load(os.path.join(self.cough_dir, cid + ".npy")),
                             dtype=torch.float32)

        c_raw = per_frequency_normalize(c_raw)
        c_scaled = pad_to_224(c_raw)
        if self.is_train and self.augmentation != "none":
            c_scaled = apply_augmentation(c_scaled, self.augmentation)

        m_speech = self._mean_speech_image(pid)
        m_speech = per_frequency_normalize(m_speech)
        m_speech = pad_to_224(m_speech)

        return torch.stack([c_scaled, c_scaled, m_speech], dim=0), label, pid


def get_early_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss,
                          batch_size, arch, num_outer_folds=10, augmentation="none"):
    train_folds = [data_folds + f"/fold_{k}.csv"
                   for k in range(num_outer_folds) if k != j and k != i]

    train_ds = ConcatDataset([
        EarlyFusionFlatDataset(f, cough_dir, speech_dir, arch,
                               is_train=True, augmentation=augmentation)
        for f in train_folds
    ])
    val_ds  = EarlyFusionFlatDataset(data_folds + f"/fold_{j}.csv",
                                     cough_dir, speech_dir, arch,
                                     is_train=False, augmentation=augmentation) \
        if j is not None else None
    test_ds = EarlyFusionFlatDataset(data_folds + f"/fold_{i}.csv",
                                     cough_dir, speech_dir, arch,
                                     is_train=False, augmentation=augmentation) \
        if i is not None else None

    g = create_generator()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True,
                              collate_fn=standard_collate,
                              worker_init_fn=seed_worker, generator=g)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=standard_collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=standard_collate) if test_ds else None
    return train_loader, val_loader, test_loader


class LateFusionDataset(Dataset):
    def __init__(self, annotations_file, cough_dir, speech_dir,
                 is_train=False, augmentation="none"):
        self.df = pd.read_csv(annotations_file)
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(
            lambda x: x.split('/')[0]
        )
        self.df = self.df[self.df['Cough_ID'].astype(str).map(
            lambda cid: os.path.exists(os.path.join(cough_dir, cid + ".npy"))
        )].reset_index(drop=True)
        self.df = self.df[self.df['patient_id'].map(
            lambda pid: os.path.exists(os.path.join(speech_dir, f"{pid}.pt"))
        )].reset_index(drop=True)

        self.cough_dir, self.speech_dir = cough_dir, speech_dir
        self.is_train, self.augmentation = is_train, augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid   = str(row['Cough_ID'])
        pid   = str(row['patient_id'])
        label = int(row['Status'])

        c_raw = torch.tensor(np.load(os.path.join(self.cough_dir, cid + ".npy")),
                             dtype=torch.float32)
        c_raw = per_frequency_normalize(c_raw)
        c_img = pad_to_224(c_raw)
        if self.is_train and self.augmentation != "none":
            c_img = apply_augmentation(c_img, self.augmentation)
        c_3ch = c_img.unsqueeze(0).repeat(3, 1, 1)

        s_raw = torch.load(os.path.join(self.speech_dir, f"{pid}.pt"),
                           weights_only=True)
        speech = s_raw.mean(dim=1)

        return c_3ch, speech, label, pid


def get_late_fusion_data(dataset, data_folds, i, j, cough_dir, speech_dir,
                         loss, batch_size, num_outer_folds=10, augmentation="none"):
    train_folds = [data_folds + f"/fold_{k}.csv"
                   for k in range(num_outer_folds) if k != j and k != i]

    train_ds = ConcatDataset([
        LateFusionDataset(f, cough_dir, speech_dir,
                          is_train=True, augmentation=augmentation)
        for f in train_folds
    ])
    val_ds  = LateFusionDataset(data_folds + f"/fold_{j}.csv", cough_dir, speech_dir,
                                is_train=False, augmentation=augmentation) \
        if j is not None else None
    test_ds = LateFusionDataset(data_folds + f"/fold_{i}.csv", cough_dir, speech_dir,
                                is_train=False, augmentation=augmentation) \
        if i is not None else None

    g = create_generator()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True,
                              collate_fn=late_fusion_collate,
                              worker_init_fn=seed_worker, generator=g)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=late_fusion_collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True,
                             collate_fn=late_fusion_collate) if test_ds else None
    return train_loader, val_loader, test_loader