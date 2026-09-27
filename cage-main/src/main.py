# ------------------------------------------------------------------
# main.py
# Runs the hyperparameter grid. For each hyperparameter combination,
# trains a model on all 10x9 fold pairs and logs every epoch's metrics
# to a CSV file named after the hyperparameters.
#
# Test AUC is only populated at the single dev-selected epoch. All
# other epochs have NaN in the test_* columns.
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

    'test_set': [0],
    'dev_set':  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'num_epochs': [50],
    'batch_size': [32],
    'learning_rate': [1e-3],
    'weight_decay': [1e-4],
    'dataset': ["cage"],
    'arch': ["resnet"],
    'fusion': ["early"],
    'augmentation': ["none","gaussian_noise","solarisation"],
    'pretrained': [True],

    'early_stop_patience':   [5],
    'early_stop_min_delta':  [5e-3],   # raised from 1e-3
    'early_stop_min_epochs': [15],
    'early_stop_window':     [5],      # new — replaces hardcoded window=3
}

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
cough_dir  = "data/cage/mel_spectrograms_128"
speech_dir = "data/cage/preprocessed_speech"


# ------------------------------------------------------------------
# Build a descriptive log filename from the hyperparameters only.
# ------------------------------------------------------------------
def build_log_filename(point):
    filename = (
        f"{point['fusion']}"
        f"_{point['dataset']}"
        f"_{point['arch']}"
        f"_pretrained-{point['pretrained']}"
        f"_{point['augmentation']}"
        f"_loss-{point['loss_selected']}"
        f"_lr{point['learning_rate']}"
        f"_wd{point['weight_decay']}"
        f"_bs{point['batch_size']}"
        f"_ep{point['num_epochs']}"
        f".csv"
    )
    return os.path.join("results", filename)


def main(grid):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("results", exist_ok=True)

    header = (
        "fusion,dataset,test_set,dev_set,arch,pretrained,augmentation,"
        "lr,wd,batch_size,"
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

        weights = (ResNet18_Weights.IMAGENET1K_V1
                   if point['pretrained'] else None)

        # ------------------------------------------------------------------
        # Build data loaders + model
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
                pretrained=point['pretrained'],
            )
            if point['arch'] == "resnet":
                model = ResNet18(fusion_type="none", num_classes=2,
                                 use_pretrained=point['pretrained']).to(device)
            elif point['arch'] == "lr":
                input_dim = 2 * 224 * 224
                model = Logistic_Regression(input_dim=input_dim,
                                            num_classes=2).to(device)
            else:
                raise ValueError(f"Unknown arch: {point['arch']}")

        elif point['fusion'] == "intermediate":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation'],
                pretrained=point['pretrained'],
            )
            model = Intermediate(num_classes=2).to(device)

        elif point['fusion'] == "none":
            train, val, test = get_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir,
                loss=point['loss_selected'],
                batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation'],
                pretrained=point['pretrained'],
            )
            if point['arch'] == "resnet":
                model = ResNet18(fusion_type="none", num_classes=2,
                                 use_pretrained=point['pretrained']).to(device)
            elif point['arch'] == "lr":
                input_dim = 128
                model = Logistic_Regression(input_dim=input_dim,
                                            num_classes=2).to(device)
            else:
                raise ValueError(f"Unknown arch: {point['arch']}")

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
        # Train and collect per-epoch history
        # ------------------------------------------------------------------
        params = dict(point)
        params["use_pretrained"] = point['pretrained']
        history = train_validate(train, val, test, model, params)

        # ------------------------------------------------------------------
        # Write every epoch of this fold to the log file
        # ------------------------------------------------------------------
        with open(log_file, "a") as f:
            if write_header:
                f.write(header)

            for h in history:
                f.write(
                    f"{point['fusion']},{point['dataset']},"
                    f"{point['test_set']},{point['dev_set']},"
                    f"{point['arch']},"
                    f"{point['pretrained']},{point['augmentation']},"
                    f"{point['learning_rate']},{point['weight_decay']},"
                    f"{point['batch_size']},"
                    f"{h['epoch']},{h['train_loss']:.4f},"
                    f"{h['dev_loss_patient']:.4f},{h['dev_loss_cough']:.4f},"
                    f"{h['dev_acc']:.4f},{h['dev_auc']:.4f},"
                    f"{h['test_loss_patient']:.4f},{h['test_loss_cough']:.4f},"
                    f"{h['test_acc']:.4f},{h['test_auc']:.4f},"
                    f"{h['is_best']}\n"
                )


if __name__ == "__main__":
    main(grid)