import itertools
import os
import random
import torch
from dataloader import *
from model_scripts import *
import torchvision 
from torchvision import models
from torchvision.models import ResNet18_Weights

grid = {
    'loss_selected': ["cross_entropy_resnet"],
    'test_set': [0,1,2,3,4,5,6,7,8,9],
    'dev_set': [0,1,2,3,4,5,6,7,8,9],
    'num_epochs': [40],
    'batch_size': [32],
    'learning_rate': [3e-4],
    'weight_decay': [1e-2],
    'dataset': ["cage"],
    'arch': ["resnet"],	#resnet or lr ->logistic regression
    'fusion': ["early"],   # "early" or "late" or "none" or "intermediate"
    'augmentation': ["time_masking"] #"gaussian_noise", "solarisation", "frequency_masking", "time_masking"
}

cough_dir = "data/cage/mel_spectrograms_128"
speech_dir = "data/cage/preprocessed_speech_224"  # per-patient .pt tensors from preprocessdata.py


def build_log_filename(point):
    """Build a log filename from the config-defining hyperparameters
    (i.e. everything except the fold indices test_set / dev_set)."""
    return os.path.join("logs", (
        f"{point['fusion']}"
        f"_{point['dataset']}"
        f"_{point['arch']}"
        f"_{point['augmentation']}"
        f"_loss-{point['loss_selected']}"
        f"_lr{point['learning_rate']}"
        f"_wd{point['weight_decay']}"
        f"_bs{point['batch_size']}"
        f"_ep{point['num_epochs']}"
        f".csv"
    ))


def main(grid):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("logs", exist_ok=True)

    header = ("fusion,dataset,test_set,dev_set,augmentation,lr,wd,"
              "batch_size,dev_acc,dev_auc,test_acc,test_auc\n")

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        if point['test_set'] == point['dev_set']:
            continue

        log_file = build_log_filename(point)
        write_header = not os.path.exists(log_file)

        if point['fusion'] == "early":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/"+point['dataset']+"/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                arch = point['arch'],
                augmentation=point['augmentation']
            )
            if point['arch'] == "resnet":
                model = ResNet18(num_classes=2,weights = ResNet18_Weights.IMAGENET1K_V1).to(device)
            elif point['arch'] == "lr":
                model = Logistic_Regression(num_classes=2,input_dim = 2*224*224).to(device)
        elif point['fusion'] == "intermediate":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/"+point['dataset']+"/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation']
            )
            model = Intermediate(num_classes=2).to(device)
        elif point['fusion'] == "none":
            train, val, test = get_data(
                dataset=point['dataset'],
                data_folds="data/"+point['dataset']+"/data_folds_filtered",   
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir,
                loss=point['loss_selected'],
                batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation']
            )
            model = ResNet18(num_classes=2).to(device)

        elif point['fusion'] == 'late':
            train, val, test = get_late_fusion_data(
                dataset=point['dataset'],
                data_folds = "data/"+point['dataset']+"/data_folds_filtered",
                i=point['test_set'],j=point['dev_set'],
                cough_dir=cough_dir,speech_dir=speech_dir,
                loss = point['loss_selected'],batch_size = point['batch_size'],num_outer_folds=10,
                augmentation = point['augmentation'],
                speech_arch = point['arch']
            )
            model = LateFusion(speech_encoding_method=point['arch'],num_classes=2).to(device)

        else:
            raise ValueError(f"Unknown fusion: {point['fusion']}")

        dev_acc, dev_auc, test_acc, test_auc = train_validate(train, val, test, model, point)

        with open(log_file, "a") as f:
            if write_header:
                f.write(header)
            f.write(f"{point['fusion']},{point['dataset']},{point['test_set']},{point['dev_set']},"
                    f"{point['augmentation']},{point['learning_rate']},{point['weight_decay']},{point['batch_size']},"
                    f"{dev_acc:.4f},{dev_auc:.4f},{test_acc:.4f},{test_auc:.4f}\n")


if __name__ == "__main__":
    main(grid)
