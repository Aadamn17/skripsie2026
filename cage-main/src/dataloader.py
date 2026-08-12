import os
import torch # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.io import wavfile # type: ignore
from torch.utils.data import Dataset, DataLoader, ConcatDataset # type: ignore
from typing import List, Tuple

cough_dir = "../data/cage/mel_spectograms_128"
speech_dir = "../data/cage/mel_spectograms_counting_128"
def get_cough_data(dataset, data_folds, i, j, cough_dir, loss, batch_size, num_outer_folds=10):
    """
    dataset : string that indicates which dataset is being loaded, ("hyfe" or "cage")
    data_folds: path to the folds that contain the patients and their label for each cross validation fold
    i: integer that indicates the fold that is held out for development
    j: integer that indicates the fold that is held out for testing   
    dir: path to mel-spectrogram directory
    loss: string that is used to load data according to the model ("cross_entropy" for LR, "cross_entropy_resnet" for Resnet)
    batch_size: integer indicating the number of coughs in one batch
    num_outer_folds: integer indicating the number of cross-validation outer folds into which the data will be divided

    Returns dataloaders with the training, development and test data
    """
    train_data = None
    val_data = None
    test_data = None

    # Get all the folds in the training set, which excludes the test and dev set folds.
    train_set_files = []    
    for k in range(num_outer_folds):
        if not k==j and not k==i:
            train_set_files.append(data_folds + "/fold_" + str(k))
            
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)
    
    # Calculate the mean and standard devation of the training dataset for dataset normalisation
    mean, std = get_mean_std_cough(train_set_files, dataset, cough_dir, 128)

    # Get all the training data using a dataloader class
    train_data_set = ConcatDataset([CoughDatasetCleaned(dataset, file + ".csv",cough_dir, loss, mean=mean, std=std) for file in train_set_files])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    
    # Get all the development/validation data using a dataloader class if development data is not None
    if not j is None:
        val_data_set = CoughDatasetCleaned(dataset, dev_set_file+".csv", cough_dir, loss, mean=mean, std=std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: val_data = None

    # Get all the test data using a dataloader class if development data is not None
    if not i is None:
        test_data_set = CoughDatasetCleaned(dataset, test_set_file+".csv", cough_dir, loss, mean=mean, std=std)
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: test_data = None

    return train_data, val_data, test_data
def get_speech_data(dataset, data_folds, i, j, speech_dir, loss, batch_size, num_outer_folds=10):
    """
    dataset : string that indicates which dataset is being loaded, ("hyfe" or "cage")
    data_folds: path to the folds that contain the patients and their label for each cross validation fold
    i: integer that indicates the fold that is held out for development
    j: integer that indicates the fold that is held out for testing   
    dir: path to mel-spectrogram directory
    loss: string that is used to load data according to the model ("cross_entropy" for LR, "cross_entropy_resnet" for Resnet)
    batch_size: integer indicating the number of coughs in one batch
    num_outer_folds: integer indicating the number of cross-validation outer folds into which the data will be divided

    Returns dataloaders with the training, development and test data
    """
    train_data = None
    val_data = None
    test_data = None

    # Get all the folds in the training set, which excludes the test and dev set folds.
    train_set_files = []    
    for k in range(num_outer_folds):
        if not k==j and not k==i:
            train_set_files.append(data_folds + "/fold_" + str(k))
            
    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)
    
    # Calculate the mean and standard devation of the training dataset for dataset normalisation
    mean, std = get_mean_std(train_set_files, dataset, speech_dir, 128)

    # Get all the training data using a dataloader class
    train_data_set = ConcatDataset([CoughDatasetCleaned(dataset, file + ".csv", speech_dir, loss, mean=mean, std=std) for file in train_set_files])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    
    # Get all the development/validation data using a dataloader class if development data is not None
    if not j is None:
        val_data_set = CoughDatasetCleaned(dataset, dev_set_file+".csv", speech_dir, loss, mean=mean, std=std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: val_data = None

    # Get all the test data using a dataloader class if development data is not None
    if not i is None:
        test_data_set = CoughDatasetCleaned(dataset, test_set_file+".csv", speech_dir, loss, mean=mean, std=std)
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: test_data = None

    return train_data, val_data, test_data
def get_mean_std_cough(train_set_files, dataset, cough_dir, inner_bins=128): 
    """
    Calculate the mean and standard deviation of the training data for dataset normalisation.
    train_set_files: list of file paths to the folds that contain the patients and their labels for all the training data
    dataset: string that indicates which dataset is being loaded, ("hyfe" or "cage")
    dir: path to mel-spectrogram directory
    inner_bins: number of frequency bins
    
    Returns the mean and standard deviation of the training dataset.
    """   
    train_data_set = EmptyDataset()
    for file in train_set_files:
        train_data_set = ConcatDataset([train_data_set, CoughDataset(dataset, file+".csv",cough_dir, inner_bins)])
        
    mels = []
    train_loader = DataLoader(train_data_set, batch_size=100, num_workers=1)
    for images, _ in train_loader:
        mels.append(images)

    mels = torch.cat(mels, dim=0)
    perc = int(2/3*mels.shape[1])
    mean = torch.mean(mels[:,:perc,:], dim=(0,1))
    std = torch.std(mels[:,:perc,:], dim=(0,1))

    return mean, std

class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")
    
class CoughDataset(Dataset):
    """
    Custom Dataset class that loads in the data and returns the plain cough images and labels
    
    dataset: string that indicates which dataset is being loaded, ("hyfe" or "cage")
    annotations_file: path to the csv file that contains the patients and their labels
    dir: path to mel-spectrogram directory
    bins: number of frequency bins
    
    Returns the mel-spectrogram image and the label for each cough recording
    """
    def __init__(self, dataset, annotations_file, cough_dir, bins=128):
        self.labels = pd.read_csv(annotations_file)
        self.dir = cough_dir
        self.dataset = dataset
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels["Status"][idx] 
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx])+".npy")
        image = torch.tensor(np.transpose(np.load(path)))
        image = image[:50,:]

        if self.dataset == "hyfe" and image.shape[0]<40: image = torch.nn.functional.pad(image, (0,0,(40-image.shape[0]),0), "constant", 0)
        if self.dataset == "cage" and image.shape[0]<50: image = torch.nn.functional.pad(image, (0,0,(50-image.shape[0]),0), "constant", 0)

        return image, label
    
class CoughDatasetCleaned(Dataset):
    """
    Custom Dataset Class that loads in the coughs as tensors, pads the frequency bins, standardises the coughs with the entire training set mean and standard deviation and returns the mean of the tensors and the labels for the LR or Resnet models.
    
    dataset: string that indicates which dataset is being loaded, ("hyfe" or "cage")
    annotations_file: path to the csv file that contains the patients and their labels
    dir: path to mel-spectrogram directory
    loss: string that is used to load data according to the model ("cross_entropy" for LR, "cross_entropy_resnet" for Resnet)
    mean: mean of the training data
    std: standard deviation of the training data
    
    Returns the mel-spectrogram image and the label for each cough recording
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
        if self.dataset == "hyfe" or self.dataset == "cage":
            label = self.labels["Status"][idx]   
 
            path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx])+".npy")
            image_raw = torch.tensor(np.transpose(np.load(path)))
            
            # Load in raw audio if needed:
            if self.dataset == "cage": audio_raw = wavfile.read(os.path.join(self.dir.replace("mel_spectrograms_128", "audio"), str(self.labels["Cough_ID"][idx]).replace("/", "/cough_")+".wav"))[1]
            elif self.dataset == "hyfe": audio_raw = wavfile.read(os.path.join(self.dir.replace("mel_spectrograms_128", "audio"), str(self.labels["Cough_ID"][idx])+"-recording-1.wav"))[1]

            if self.dataset == "cage" and image_raw.shape[0]<40: image_raw = torch.nn.functional.pad(image_raw, (0,0,(40-image_raw.shape[0]),0), "constant", image_raw.min())
            if self.dataset == "hyfe" and image_raw.shape[0]<40: image_raw = torch.nn.functional.pad(image_raw, (0,0,(40-image_raw.shape[0]),0), "constant", image_raw.min())

            if self.loss == "cross_entropy": 
                image = self.standardize(image_raw)
                image_mean = image.mean(0)
                return image_mean, label
            elif self.loss == 'cross_entropy_resnet':
                return self.pad(self.standardize(self.repeat(image_raw))), label

    def repeat(self, image):
        image = image[None, :, :]
        image = torch.concat((image, image, image))
        return image

    def pad(self,image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

    def standardize(self, image):
        image = (image - self.mean)/self.std
        return image

class SpeechDataset(Dataset):
    """
    Custom Dataset class that loads in the data and returns the plain speech images and labels
    
    dataset: string that indicates which dataset is being loaded, ("hyfe" or "cage")
    annotations_file: path to the csv file that contains the patients and their labels
    dir: path to mel-spectrogram directory
    bins: number of frequency bins
    
    Returns the mel-spectrogram image and the label for each speech recording
    """
    def __init__(self, dataset, annotations_file, speech_dir, bins=128):
        self.labels = pd.read_csv(annotations_file)
        self.dir = speech_dir
        self.dataset = dataset
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels["Status"][idx] 
        path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx])+".npy")
        image = torch.tensor(np.transpose(np.load(path)))
        image = image[:50,:]

        if self.dataset == "hyfe" and image.shape[0]<40: image = torch.nn.functional.pad(image, (0,0,(40-image.shape[0]),0), "constant", 0)
        if self.dataset == "cage" and image.shape[0]<50: image = torch.nn.functional.pad(image, (0,0,(50-image.shape[0]),0), "constant", 0)

        return image, label
    
class speechDatasetCleaned(Dataset):
    """
    Custom Dataset Class that loads in the speech as tensors, pads the frequency bins, standardises the speech with the entire training set mean and standard deviation and returns the mean of the tensors and the labels for the LR or Resnet models.
    
    dataset: string that indicates which dataset is being loaded, ("hyfe" or "cage")
    annotations_file: path to the csv file that contains the patients and their labels
    dir: path to mel-spectrogram directory
    loss: string that is used to load data according to the model ("cross_entropy" for LR, "cross_entropy_resnet" for Resnet)
    mean: mean of the training data
    std: standard deviation of the training data
    
    Returns the mel-spectrogram image and the label for each speech recording
    """
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
        if self.dataset == "hyfe" or self.dataset == "cage":
            label = self.labels["Status"][idx]   
 
            path = os.path.join(self.dir, str(self.labels["Cough_ID"][idx])+".npy")
            image_raw = torch.tensor(np.transpose(np.load(path)))
            
            # Load in raw audio if needed:
            if self.dataset == "cage": audio_raw = wavfile.read(os.path.join(self.dir.replace("mel_spectrograms_128", "audio"), str(self.labels["Cough_ID"][idx]).replace("/", "/cough_")+".wav"))[1]
            elif self.dataset == "hyfe": audio_raw = wavfile.read(os.path.join(self.dir.replace("mel_spectrograms_128", "audio"), str(self.labels["Cough_ID"][idx])+"-recording-1.wav"))[1]

            if self.dataset == "cage" and image_raw.shape[0]<40: image_raw = torch.nn.functional.pad(image_raw, (0,0,(40-image_raw.shape[0]),0), "constant", image_raw.min())
            if self.dataset == "hyfe" and image_raw.shape[0]<40: image_raw = torch.nn.functional.pad(image_raw, (0,0,(40-image_raw.shape[0]),0), "constant", image_raw.min())

            if self.loss == "cross_entropy": 
                image = self.standardize(image_raw)
                image_mean = image.mean(0)
                return image_mean, label
            elif self.loss == 'cross_entropy_resnet':
                return self.pad(self.standardize(self.repeat(image_raw))), label

    def repeat(self, image):
        image = image[None, :, :]
        image = torch.concat((image, image, image))
        return image

    def pad(self,image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0), (0, 224 - image.shape[-2]), (0, 224 - image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

    def standardize(self, image):
        image = (image - self.mean)/self.std
        return image


class FusedCoughSpeechDataset(Dataset):
    """
    Loads both cough and speech mel-spectrograms for the same Cough_ID,
    standardises each with its own training mean/std, pools over time,
    and concatenates the two 128-dim vectors into a single 256-dim vector.
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

        # ----- Load and preprocess cough -----
        cough_path = os.path.join(self.cough_dir, cough_id + ".npy")
        cough_raw = torch.tensor(np.transpose(np.load(cough_path)))
        # Standardise, then pool over time (dim=0)
        cough_norm = (cough_raw - self.cough_mean) / self.cough_std
        cough_pooled = cough_norm.mean(0)   # shape: (128,)

        # ----- Load and preprocess speech -----
        speech_path = os.path.join(self.speech_dir, cough_id + ".npy")
        speech_raw = torch.tensor(np.transpose(np.load(speech_path)))
        speech_norm = (speech_raw - self.speech_mean) / self.speech_std
        speech_pooled = speech_norm.mean(0) # shape: (128,)

        # ----- Fuse: concatenate along feature dimension -----
        fused = torch.cat([cough_pooled, speech_pooled])  # shape: (256,)

        return fused, label
def get_fused_mean_std(train_set_files, dataset, cough_dir, speech_dir, bins=128):
    """
    Computes separate mean/std for cough and speech pooled features
    over the training set.
    """
    # We'll collect pooled features for each modality
    cough_feats, speech_feats = [], []

    for file in train_set_files:
        labels = pd.read_csv(file + ".csv")
        for _, row in labels.iterrows():
            cough_id = str(row["Cough_ID"])
            
            # Cough
            cough_path = os.path.join(cough_dir, cough_id + ".npy")
            cough_img = torch.tensor(np.transpose(np.load(cough_path)))
            cough_pooled = cough_img.mean(0)   # time average
            cough_feats.append(cough_pooled)
            
            # Speech
            speech_path = os.path.join(speech_dir, cough_id + ".npy")
            speech_img = torch.tensor(np.transpose(np.load(speech_path)))
            speech_pooled = speech_img.mean(0)
            speech_feats.append(speech_pooled)

    cough_feats = torch.stack(cough_feats)    # dim->(N, 128)
    speech_feats = torch.stack(speech_feats)  # dim->(N, 128)

    cough_mean = cough_feats.mean(0)
    cough_std  = cough_feats.std(0)
    speech_mean = speech_feats.mean(0)
    speech_std  = speech_feats.std(0)

    return cough_mean, cough_std, speech_mean, speech_std

def get_fused_data(dataset, data_folds, i, j, cough_dir, speech_dir, loss, batch_size, num_outer_folds=10):
    train_set_files = []
    for k in range(num_outer_folds):
        if k != j and k != i:
            train_set_files.append(data_folds + "/fold_" + str(k))

    dev_set_file = data_folds + "/fold_" + str(j)
    test_set_file = data_folds + "/fold_" + str(i)

    # Compute means/stds from training folds
    cough_mean, cough_std, speech_mean, speech_std = get_fused_mean_std(
        train_set_files, dataset, cough_dir, speech_dir
    )

    # Training set
    train_data_set = ConcatDataset([
        FusedCoughSpeechDataset(dataset, f + ".csv", cough_dir, speech_dir, loss,
                                cough_mean, cough_std, speech_mean, speech_std)
        for f in train_set_files
    ])
    train_data = DataLoader(train_data_set, batch_size=batch_size, 
                            num_workers=4, shuffle=True, drop_last=True)

    # Validation set
    val_data = None
    if j is not None:
        val_data_set = FusedCoughSpeechDataset(dataset, dev_set_file + ".csv", 
                                               cough_dir, speech_dir, loss,
                                               cough_mean, cough_std, speech_mean, speech_std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, 
                              num_workers=4, shuffle=True, drop_last=True)

    # Test set
    test_data = None
    if i is not None:
        test_data_set = FusedCoughSpeechDataset(dataset, test_set_file + ".csv",
                                                cough_dir, speech_dir, loss,
                                                cough_mean, cough_std, speech_mean, speech_std)
        test_data = DataLoader(test_data_set, batch_size=batch_size,
                               num_workers=4, shuffle=True, drop_last=True)

    return train_data, val_data, test_data