# ------------------------------------------------------------------
# main.py
# Runs the hyperparameter grid. For each hyperparameter combination,
# trains a model on all 10×9 fold pairs and logs every single epoch's
# metrics (train_loss, dev_loss_patient, dev_loss_cough, dev_acc,
# dev_auc, test_loss_patient, test_loss_cough, test_acc, test_auc,
# is_best) to a CSV file named after the hyperparameters.
# ------------------------------------------------------------------
import itertools
import os
import random
import torch
from dataloader import *
from model_scripts import *
from torchvision.models import ResNet18_Weights


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
    'fusion': ["early", "none"],        # "early" | "late" | "none" | "intermediate"
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
# Build a descriptive log filename from the hyperparameters only.
# Every fold pair (test_set, dev_set) for the same hyperparameters
# will append to the SAME file. Fold IDs are stored per row.
# ------------------------------------------------------------------
def build_log_filename(point):
    # The LR baseline uses a different input path than ResNet models,
    # so tag its filename with the loss it actually uses.
    if point['fusion'] == "none" and point['arch'] == "lr":
        loss_tag = "cross_entropy"
    else:
        loss_tag = point['loss_selected']

    filename = (
        f"{point['fusion']}"
        f"_{point['dataset']}"
        f"_{point['arch']}"
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
        "fusion,dataset,test_set,dev_set,augmentation,lr,wd,batch_size,"
        "epoch,train_loss,"
        "dev_loss_patient,dev_loss_cough,dev_acc,dev_auc,"
        "test_loss_patient,test_loss_cough,test_acc,test_auc,is_best\n"
    )

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))

        if point['test_set'] == point['dev_set']:
            continue

        log_file     = build_log_filename(point)
        write_header = not os.path.exists(log_file)

        # ------------------------------------------------------------------
        # 1. Build the data loaders and model for this grid point
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
                                 weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
            elif point['arch'] == "lr":
                model = Logistic_Regression(input_dim=2 * 224 * 224,
                                            num_classes=2).to(device)
            else:
                raise ValueError(f"Unknown arch for early fusion: {point['arch']}")

        elif point['fusion'] == "intermediate":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation'],
            )
            model = Intermediate(num_classes=2).to(device)

        elif point['fusion'] == "none":
            if point['arch'] == "lr":
                # LR baseline: per-bin standardize -> time-average -> [128] vector
                train, val, test = get_data(
                    dataset=point['dataset'],
                    data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                    i=point['test_set'], j=point['dev_set'],
                    cough_dir=cough_dir,
                    loss="cross_entropy",             # LR-specific input path
                    batch_size=point['batch_size'],
                    num_outer_folds=10,
                    augmentation="none",              # no augmentation for LR
                )
                model = Logistic_Regression(
                    fusion_type="none",
                    input_dim=128,                    # [128] time-averaged vector
                    num_classes=2,
                ).to(device)
            else:
                # Cough-only ResNet18 baseline
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
                model = ResNet18(fusion_type="none", num_classes=2).to(device)

        elif point['fusion'] == 'late':
            train, val, test = get_late_fusion_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation'],
                speech_arch=point['arch'],
            )
            model = LateFusion(speech_encoding_method=point['arch'],
                               num_classes=2).to(device)

        else:
            raise ValueError(f"Unknown fusion: {point['fusion']}")

        # ------------------------------------------------------------------
        # 2. Train and collect full per-epoch history
        # ------------------------------------------------------------------
        history = train_validate(train, val, test, model, point)

        # ------------------------------------------------------------------
        # 3. Write every epoch of this fold to the log file
        # ------------------------------------------------------------------
        with open(log_file, "a") as f:
            if write_header:
                f.write(header)

            for h in history:
                f.write(
                    f"{point['fusion']},{point['dataset']},"
                    f"{point['test_set']},{point['dev_set']},"
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