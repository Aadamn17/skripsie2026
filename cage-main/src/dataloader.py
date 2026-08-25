import os
import torch # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.io import wavfile # type: ignore
from torch.utils.data import Dataset, DataLoader, ConcatDataset # type: ignore

def get_data(dataset, data_folds, i, j, dir, loss, batch_size, num_outer_folds=10):
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
    mean, std = get_mean_std(train_set_files, dataset, dir, 128)

    # Get all the training data using a dataloader class
    train_data_set = ConcatDataset([CoughDatasetCleaned(dataset, file + ".csv", dir, loss, mean=mean, std=std) for file in train_set_files])
    train_data = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    
    # Get all the development/validation data using a dataloader class if development data is not None
    if not j is None:
        val_data_set = CoughDatasetCleaned(dataset, dev_set_file+".csv", dir, loss, mean=mean, std=std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: val_data = None

    # Get all the test data using a dataloader class if development data is not None
    if not i is None:
        test_data_set = CoughDatasetCleaned(dataset, test_set_file+".csv", dir, loss, mean=mean, std=std)
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: test_data = None

    return train_data, val_data, test_data

def get_mean_std(train_set_files, dataset, dir, inner_bins=128): 
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
        train_data_set = ConcatDataset([train_data_set, CoughDataset(dataset, file+".csv", dir, inner_bins)])
        
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
    def __init__(self, dataset, annotations_file, dir, bins=128):
        self.labels = pd.read_csv(annotations_file)
        self.dir = dir
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
    def __init__(self, dataset, annotations_file, dir, loss, mean, std):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = dir
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
    Custom Dataset Class that loads in the speech as tensors, pads the frequency bins, standardises the speech with the entire training set mean and standard deviation and returns the mean of the tensors and the labels for the LR or Resnet models.
    
    dataset: string that indicates which dataset is being loaded, ("hyfe" or "cage")
    annotations_file: path to the csv file that contains the patients and their labels
    dir: path to mel-spectrogram directory
    loss: string that is used to load data according to the model ("cross_entropy" for LR, "cross_entropy_resnet" for Resnet)
    mean: mean of the training data
    std: standard deviation of the training data
    
    Returns the mel-spectrogram image and the label for each cough recording
    """
    def __init__(self, dataset, annotations_file, dir, loss, mean, std):
        self.labels = pd.read_csv(annotations_file)
        self.dataset = dataset
        self.dir = dir
        self.loss = loss
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.dataset == "hyfe" or self.dataset == "cage":
            label = self.labels["Status"][idx]   
            image_list = []
            path = os.path.join(self.dir, str(self.labels["Speech_ID"][idx])+".npy")
            for images in np.load(path, allow_pickle=True):
                images = self.standardize(raw_images = torch.tensor(np.transpose(images))) #same filter banks but images have different lengths for each utterance, which doesnt matter since we will be averaging the images for each patient
                image_list.append(images) #list of standardized images for each patient
            final_patient_image = torch.cat(image_list, dim=1) #get one large tensor for each patient with all the coughs concatenated in the dim=1
            final_patient_image_mean = final_patient_image.mean(0)

            return final_patient_image_mean, label
        
    def standardize(self, raw_images):
        standardized_images = (raw_images - self.mean)/self.std
        return standardized_images
    

            