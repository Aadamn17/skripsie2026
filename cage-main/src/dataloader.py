import os
import torch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F


def _filter_existing_files(df, base_dir, id_col="Cough_ID"):
    if df.empty:
        return df.copy()
    existing_mask = df[id_col].astype(str).map(
        lambda cid: os.path.exists(os.path.join(base_dir, cid + ".npy"))
    )
    return df.loc[existing_mask].reset_index(drop=True)


# ======================================================================
# 1. Helper functions: compute mean & std per frequency bin
# ======================================================================
def get_mean_std_cough(train_set_files, dataset, cough_dir, inner_bins=128):
    train_data_set = EmptyDataset()
    for file in train_set_files:
        train_data_set = ConcatDataset([train_data_set, CoughDataset(dataset, file+".csv", cough_dir, inner_bins)])
    mels = []
    train_loader = DataLoader(train_data_set, batch_size=100, num_workers=1)
    for images, _ in train_loader:
        mels.append(images)
    mels = torch.cat(mels, dim=0)
    perc = int(2/3 * mels.shape[1])
    mean = torch.mean(mels[:,:perc,:], dim=(0,1))
    std = torch.std(mels[:,:perc,:], dim=(0,1))
    return mean, std

def get_mean_std_speech(train_set_files, dataset, speech_dir, inner_bins=128):
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
# 2. EmptyDataset
# ======================================================================
class EmptyDataset(Dataset):
    def __init__(self): pass
    def __len__(self): return 0
    def __getitem__(self, index): raise IndexError("Empty dataset cannot be indexed")

# ======================================================================
# 3. Raw dataset classes
# ======================================================================
class CoughDataset(Dataset):
    def __init__(self, dataset, annotations_file, cough_dir, bins=128):
        self.dir = cough_dir
        self.dataset = dataset
        self.labels = _filter_existing_files(pd.read_csv(annotations_file), self.dir)

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

class SpeechDataset(Dataset):
    def __init__(self, dataset, annotations_file, speech_dir, bins=128):
        self.dir = speech_dir
        self.dataset = dataset
        self.labels = _filter_existing_files(pd.read_csv(annotations_file), self.dir)

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
# 4. Cleaned dataset classes (single-modality)
# ======================================================================
class CoughDatasetCleaned(Dataset):
    def __init__(self, dataset, annotations_file, cough_dir, loss, mean, std):
        self.dataset = dataset
        self.dir = cough_dir
        self.labels = _filter_existing_files(pd.read_csv(annotations_file), self.dir)
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
            return image.mean(0), label
        elif self.loss == 'cross_entropy_resnet':
            image = self.repeat(image)
            image = self.pad(image)
            return image, label

    def repeat(self, image):
        image = image[None, :, :]
        return torch.concat((image, image, image), dim=0)

    def pad(self, image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

class SpeechDatasetCleaned(Dataset):
    def __init__(self, dataset, annotations_file, speech_dir, loss, mean, std):
        self.dataset = dataset
        self.dir = speech_dir
        self.labels = _filter_existing_files(pd.read_csv(annotations_file), self.dir)
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
            return image.mean(0), label
        elif self.loss == 'cross_entropy_resnet':
            image = self.repeat(image)
            image = self.pad(image)
            return image, label

    def repeat(self, image):
        image = image[None, :, :]
        return torch.concat((image, image, image), dim=0)

    def pad(self, image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

# ======================================================================
# 5. Patient dataset for INTERMEDIATE FUSION (Separate encoders)
#    Returns: cough_images (list), speech_images (list), label
# ======================================================================
class PatientIntermediateDataset(Dataset):
    """
    Returns lists of 3×224×224 images for ALL coughs and ALL speeches per patient.
    Used with PatientIntermediateClassifier.
    """
    def __init__(self, annotations_file, cough_dir, speech_dir,
                 cough_mean, cough_std, speech_mean, speech_std):
        self.df = pd.read_csv(annotations_file)
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(lambda x: x.split('/')[0])
        self.df = self.df[self.df['Cough_ID'].astype(str).map(
            lambda cid: os.path.exists(os.path.join(cough_dir, cid + ".npy"))
        )].reset_index(drop=True)
        self.patients = self.df.groupby('patient_id').agg({
            'Cough_ID': list,
            'Status': 'first'
        }).reset_index()
        self.cough_dir = cough_dir
        self.speech_dir = speech_dir
        self.cough_mean = cough_mean
        self.cough_std = cough_std
        self.speech_mean = speech_mean
        self.speech_std = speech_std

    def __len__(self):
        return len(self.patients)

    def _to_image(self, tensor):
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img = tensor[0, 0, :, :]
        return img.unsqueeze(0).repeat(3, 1, 1)

    def __getitem__(self, idx):
        row = self.patients.iloc[idx]
        patient_id = row['patient_id']
        label = row['Status']

        cough_images = []
        for cid in row['Cough_ID']:
            path = os.path.join(self.cough_dir, str(cid) + ".npy")
            if not os.path.exists(path):
                continue
            raw = torch.tensor(np.transpose(np.load(path)))
            norm = (raw - self.cough_mean) / self.cough_std
            cough_images.append(self._to_image(norm))

        speech_images = []
        speech_folder = os.path.join(self.speech_dir, patient_id)
        if os.path.isdir(speech_folder):
            for fname in os.listdir(speech_folder):
                if fname.endswith('.npy'):
                    path = os.path.join(speech_folder, fname)
                    raw = torch.tensor(np.transpose(np.load(path)))
                    norm = (raw - self.speech_mean) / self.speech_std
                    speech_images.append(self._to_image(norm))

        return cough_images, speech_images, label

# ======================================================================
# 6. Patient dataset for TRUE EARLY FUSION (Single encoder)
#    Returns: fused_images (list of 3×224×224 per pair), label
# ======================================================================
class PatientEarlyImageDataset(Dataset):
    """
    Pairs coughs and speeches up to the minimum count.
    For each pair: stacks [cough, speech, (cough+speech)/2] -> (3,224,224).
    Returns a LIST of these fused images per patient.
    """
    def __init__(self, annotations_file, cough_dir, speech_dir,
                 cough_mean, cough_std, speech_mean, speech_std):
        self.df = pd.read_csv(annotations_file)
        self.df['patient_id'] = self.df['Cough_ID'].astype(str).apply(lambda x: x.split('/')[0])
        self.df = self.df[self.df['Cough_ID'].astype(str).map(
            lambda cid: os.path.exists(os.path.join(cough_dir, cid + ".npy"))
        )].reset_index(drop=True)
        self.patients = self.df.groupby('patient_id').agg({
            'Cough_ID': list,
            'Status': 'first'
        }).reset_index()
        self.cough_dir = cough_dir
        self.speech_dir = speech_dir
        self.cough_mean = cough_mean
        self.cough_std = cough_std
        self.speech_mean = speech_mean
        self.speech_std = speech_std

    def __len__(self):
        return len(self.patients)

    def _to_image(self, tensor):
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return tensor[0, 0, :, :]  # (224,224)

    def __getitem__(self, idx):
        row = self.patients.iloc[idx]
        patient_id = row['patient_id']
        label = row['Status']
        cough_ids = sorted(
            [cid for cid in row['Cough_ID'] if os.path.exists(os.path.join(self.cough_dir, str(cid) + ".npy"))],
            key=lambda x: int(str(x).split('/')[-1])
        )

        # Get sorted speech files
        speech_folder = os.path.join(self.speech_dir, patient_id)
        speech_files = []
        if os.path.isdir(speech_folder):
            speech_files = sorted([f.replace('.npy', '') for f in os.listdir(speech_folder) if f.endswith('.npy')],
                                  key=lambda x: int(x))

        # Pair them
        n_pairs = min(len(cough_ids), len(speech_files))
        fused_images = []
        for i in range(n_pairs):
            cid = cough_ids[i]
            sid = speech_files[i]

            c_path = os.path.join(self.cough_dir, cid + ".npy")
            c_raw = torch.tensor(np.transpose(np.load(c_path)))
            c_norm = (c_raw - self.cough_mean) / self.cough_std
            c_img = self._to_image(c_norm)

            s_path = os.path.join(self.speech_dir, patient_id, sid + ".npy")
            s_raw = torch.tensor(np.transpose(np.load(s_path)))
            s_norm = (s_raw - self.speech_mean) / self.speech_std
            s_img = self._to_image(s_norm)

            stacked = torch.stack([c_img, s_img, (c_img + s_img) / 2], dim=0)  # (3,224,224)
            fused_images.append(stacked)

        return fused_images, label

# ======================================================================
# 7. Loader functions
# ======================================================================
def get_cough_data(dataset, data_folds, i, j, cough_dir, loss, batch_size, num_outer_folds=10):
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)
    mean, std = get_mean_std_cough(train_set_files, dataset, cough_dir, 128)
    train_data = DataLoader(ConcatDataset([CoughDatasetCleaned(dataset, f+".csv", cough_dir, loss, mean, std) for f in train_set_files]), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_data = DataLoader(CoughDatasetCleaned(dataset, dev_set_file+".csv", cough_dir, loss, mean, std), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True) if j is not None else None
    test_data = DataLoader(CoughDatasetCleaned(dataset, test_set_file+".csv", cough_dir, loss, mean, std), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True) if i is not None else None
    return train_data, val_data, test_data

def get_speech_data(dataset, data_folds, i, j, speech_dir, loss, batch_size, num_outer_folds=10):
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)
    mean, std = get_mean_std_speech(train_set_files, dataset, speech_dir, 128)
    train_data = DataLoader(ConcatDataset([SpeechDatasetCleaned(dataset, f+".csv", speech_dir, loss, mean, std) for f in train_set_files]), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_data = DataLoader(SpeechDatasetCleaned(dataset, dev_set_file+".csv", speech_dir, loss, mean, std), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True) if j is not None else None
    test_data = DataLoader(SpeechDatasetCleaned(dataset, test_set_file+".csv", speech_dir, loss, mean, std), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True) if i is not None else None
    return train_data, val_data, test_data

def get_intermediate_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size=1, num_outer_folds=10):
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)
    c_mean, c_std = get_mean_std_cough(train_set_files, dataset, cough_dir, 128)
    s_mean, s_std = get_mean_std_speech(train_set_files, dataset, speech_dir, 128)
    train_data = DataLoader(ConcatDataset([PatientIntermediateDataset(f+".csv", cough_dir, speech_dir, c_mean, c_std, s_mean, s_std) for f in train_set_files]), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_data = DataLoader(PatientIntermediateDataset(dev_set_file+".csv", cough_dir, speech_dir, c_mean, c_std, s_mean, s_std), batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True) if j is not None else None
    test_data = DataLoader(PatientIntermediateDataset(test_set_file+".csv", cough_dir, speech_dir, c_mean, c_std, s_mean, s_std), batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True) if i is not None else None
    return train_data, val_data, test_data

def get_early_image_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size=1, num_outer_folds=10):
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)
    c_mean, c_std = get_mean_std_cough(train_set_files, dataset, cough_dir, 128)
    s_mean, s_std = get_mean_std_speech(train_set_files, dataset, speech_dir, 128)
    train_data = DataLoader(ConcatDataset([PatientEarlyImageDataset(f+".csv", cough_dir, speech_dir, c_mean, c_std, s_mean, s_std) for f in train_set_files]), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_data = DataLoader(PatientEarlyImageDataset(dev_set_file+".csv", cough_dir, speech_dir, c_mean, c_std, s_mean, s_std), batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True) if j is not None else None
    test_data = DataLoader(PatientEarlyImageDataset(test_set_file+".csv", cough_dir, speech_dir, c_mean, c_std, s_mean, s_std), batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True) if i is not None else None
    return train_data, val_data, test_data