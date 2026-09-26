# ------------------------------------------------------------------
# main.py
# Grid search over hyperparameters. Trains each (fusion, arch,
# use_pretrained, augmentation, ...) combination on all 10×9 fold
# pairs, logging every epoch to a CSV per hyperparameter combination.
# ------------------------------------------------------------------
import itertools
import os
import random
import torch
from dataloader import *
from model_scripts import *


# ------------------------------------------------------------------
# Hyperparameter grid
# ------------------------------------------------------------------
grid = {
    'loss_selected': ["cross_entropy_resnet"],
    'test_set': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'dev_set':  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'num_epochs': [40],
    'batch_size': [32],
    'learning_rate': [1e-4],
    'weight_decay': [1e-2],
    'dataset': ["cage"],
    'arch': ["resnet", "lr"],           # "resnet" | "lr"
    'fusion': ["early", "none"],        # "early" | "none"
    'use_pretrained': [True, False],    # only affects ResNet configurations
    'augmentation': ["time_masking"],

    # Early stopping (AUC selection, smoothed, with min-epochs guard)
    'early_stop_patience':   [7],
    'early_stop_min_delta':  [1e-3],
    'early_stop_min_epochs': [10],
}

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
cough_dir  = "data/cage/mel_spectrograms_128"
speech_dir = "data/cage/preprocessed_speech"


# ------------------------------------------------------------------
# Log filename
# ------------------------------------------------------------------
def build_log_filename(point):
    if point['fusion'] == "none" and point['arch'] == "lr":
        loss_tag = "cross_entropy"
    else:
        loss_tag = point['loss_selected']

    filename = (
        f"{point['fusion']}"
        f"_{point['dataset']}"
        f"_{point['arch']}"
        f"_pretrained-{point['use_pretrained']}"
        f"_{point['augmentation']}"
        f"_loss-{loss_tag}"
        f"_lr{point['learning_rate']}"
        f"_wd{point['weight_decay']}"
        f"_bs{point['batch_size']}"
        f"_ep{point['num_epochs']}"
        f".csv"
    )
    return os.path.join("logs", filename)


def main(grid):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("logs", exist_ok=True)

    header = (
        "fusion,dataset,test_set,dev_set,arch,use_pretrained,augmentation,"
        "lr,wd,batch_size,"
        "epoch,train_loss,"
        "dev_loss_patient,dev_loss_cough,dev_acc,dev_auc,"
        "test_loss_patient,test_loss_cough,test_acc,test_auc,is_best\n"
    )

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))

        if point['test_set'] == point['dev_set']:
            continue

        # LR baseline does not use pretrained weights -> skip duplicate runs
        if point['arch'] == "lr" and point['use_pretrained'] is False:
            continue

        log_file     = build_log_filename(point)
        write_header = not os.path.exists(log_file)

        # ------------------------------------------------------------------
        # 1. Build data loaders and model
        # ------------------------------------------------------------------
        if point['fusion'] == "early":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                arch=point['arch'],
                augmentation=point['augmentation'],
            )
            if point['arch'] == "resnet":
                model = ResNet18(fusion_type="none", num_classes=2,
                                 use_pretrained=point['use_pretrained']).to(device)
            elif point['arch'] == "lr":
                model = Logistic_Regression(
                    fusion_type="none",
                    input_dim=2 * 224 * 224,
                    num_classes=2,
                ).to(device)
            else:
                raise ValueError(f"Unknown arch for early fusion: {point['arch']}")

        elif point['fusion'] == "none":
            if point['arch'] == "lr":
                # LR baseline: time-averaged raw log-mel -> [128]
                train, val, test = get_data(
                    dataset=point['dataset'],
                    data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                    i=point['test_set'], j=point['dev_set'],
                    cough_dir=cough_dir,
                    loss="cross_entropy",
                    batch_size=point['batch_size'],
                    num_outer_folds=10,
                    augmentation="none",
                )
                model = Logistic_Regression(
                    fusion_type="none",
                    input_dim=128,
                    num_classes=2,
                ).to(device)
            else:
                # Cough-only ResNet-18 baseline
                train, val, test = get_data(
                    dataset=point['dataset'],
                    data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                    i=point['test_set'], j=point['dev_set'],
                    cough_dir=cough_dir,
                    loss=point['loss_selected'],
                    batch_size=point['batch_size'],
                    num_outer_folds=10,
                    augmentation=point['augmentation'],
                )
                model = ResNet18(fusion_type="none", num_classes=2,
                                 use_pretrained=point['use_pretrained']).to(device)

        else:
            raise ValueError(f"Unknown fusion: {point['fusion']}")

        # ------------------------------------------------------------------
        # 2. Train
        # ------------------------------------------------------------------
        history = train_validate(train, val, test, model, point)

        # ------------------------------------------------------------------
        # 3. Write per-epoch history
        # ------------------------------------------------------------------
        with open(log_file, "a") as f:
            if write_header:
                f.write(header)

            for h in history:
                f.write(
                    f"{point['fusion']},{point['dataset']},"
                    f"{point['test_set']},{point['dev_set']},"
                    f"{point['arch']},{point['use_pretrained']},"
                    f"{point['augmentation']},{point['learning_rate']},"
                    f"{point['weight_decay']},{point['batch_size']},"
                    f"{h['epoch']},{h['train_loss']:.4f},"
                    f"{h['dev_loss_patient']:.4f},{h['dev_loss_cough']:.4f},"
                    f"{h['dev_acc']:.4f},{h['dev_auc']:.4f},"
                    f"{h['test_loss_patient']:.4f},{h['test_loss_cough']:.4f},"
                    f"{h['test_acc']:.4f},{h['test_auc']:.4f},"
                    f"{h['is_best']}\n"
                )


if __name__ == "__main__":
    main(grid)