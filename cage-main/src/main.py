import glob
import itertools
import os
import random
import torch
import numpy as np
import pandas as pd
from dataloader import *
from model_scripts import *

# Anchor output paths to this file's directory
PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

grid = {
    'loss_selected':          ["cross_entropy_resnet"],
    'test_set':               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'dev_set':                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'num_epochs':             [50],
    'batch_size':             [32],
    'learning_rate':          [1e-4, 1e-5],
    'weight_decay':           [1e-4],
    'dataset':                ["cage"],
    'arch':                   ["resnet", "lr"],
    'fusion':                 ["none", "early", "late"],
    'augmentation':           ["none", "solarisation"],
    'use_pretrained':         [True, False],
    'early_stop_patience':    [5],
    'early_stop_min_delta':   [5e-3],
    'early_stop_min_epochs':  [15],
    'early_stop_window':      [5],
}

raw_speech_dir = "data/cage/raw_speech"
cough_dir      = "data/cage/mel_spectrograms_128"
speech_dir     = "data/cage/preprocessed_speech"

LR_BATCH_SIZE = 256

KEY_COLUMNS = ["fusion", "arch", "test_set", "dev_set",
               "lr", "wd", "augmentation", "use_pretrained"]


def build_log_filename(point):
    """One CSV per config. All 90 fold pairs append rows into this file."""
    return os.path.join(RESULTS_DIR, (
        f"fusion-{point['fusion']}_dataset-{point['dataset']}_arch-{point['arch']}_"
        f"pretrained-{str(point['use_pretrained']).lower()}_aug-{point['augmentation']}_"
        f"lr-{point['learning_rate']}_wd-{point['weight_decay']}.csv"
    ))


def point_key(point):
    """A hashable key that identifies one (config, fold pair)."""
    return (
        str(point["fusion"]),
        str(point["arch"]),
        int(point["test_set"]),
        int(point["dev_set"]),
        float(point["learning_rate"]),
        float(point["weight_decay"]),
        str(point["augmentation"]),
        bool(point["use_pretrained"]),
    )


def load_completed_keys(results_dir):
    """Scan every existing results CSV and return the set of completed
    (config, test_set, dev_set) tuples."""
    completed = set()
    if not os.path.isdir(results_dir):
        return completed
    for f in glob.glob(os.path.join(results_dir, "*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not set(KEY_COLUMNS).issubset(df.columns):
            continue
        for _, row in df.iterrows():
            try:
                completed.add((
                    str(row["fusion"]),
                    str(row["arch"]),
                    int(row["test_set"]),
                    int(row["dev_set"]),
                    float(row["lr"]),
                    float(row["wd"]),
                    str(row["aug"]),
                    bool(row["use_pretrained"]),
                ))
            except (ValueError, TypeError):
                continue
    return completed


def main(grid):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Load any completed work so a restart does not redo it
    completed = load_completed_keys(RESULTS_DIR)
    print(f"[resume] {len(completed)} (config, fold pair) combinations already completed")

    header = (
        "fusion,dataset,test_set,dev_set,arch,use_pretrained,augmentation,"
        "lr,wd,batch_size,epoch,train_loss,"
        "dev_loss_patient,dev_loss_cough,dev_acc,dev_auc,dev_sens,dev_spec,"
        "test_loss_patient,test_loss_cough,test_acc,test_auc,test_sens,test_spec,is_best\n"
    )

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        if point['test_set'] == point['dev_set']:
            continue
        if point['arch'] == "lr" and point['fusion'] != "none":
            continue
        if point['arch'] == "lr" and point['use_pretrained'] is False:
            continue

        # Skip only if this exact (config, fold pair) is already done
        if point_key(point) in completed:
            continue

        effective_loss       = "cross_entropy" if point['arch'] == "lr" else point['loss_selected']
        effective_batch_size = LR_BATCH_SIZE    if point['arch'] == "lr" else point['batch_size']

        if point['fusion'] == "early":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=effective_loss, batch_size=effective_batch_size,
                arch=point['arch'], augmentation=point['augmentation'],
            )
            model = ResNet18(num_classes=2,
                             use_pretrained=point['use_pretrained']).to(device)

        elif point['fusion'] == "none":
            train, val, test = get_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir,
                loss=effective_loss,
                batch_size=effective_batch_size,
                augmentation=point['augmentation'],
            )
            if point['arch'] == "lr":
                model = Logistic_Regression(input_dim=128, num_classes=2).to(device)
            elif point['arch'] == "resnet":
                model = ResNet18(num_classes=2,
                                 use_pretrained=point['use_pretrained']).to(device)
            else:
                raise ValueError(f"Unknown arch: {point['arch']}")

        elif point['fusion'] == 'late':
            train, val, test = get_late_fusion_data(
                dataset=point['dataset'],
                data_folds="data/" + point['dataset'] + "/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=effective_loss, batch_size=effective_batch_size,
                augmentation=point['augmentation'],
            )
            model = LateFusion(
                num_classes=2,
                use_pretrained=point['use_pretrained'],
            ).to(device)

        else:
            raise ValueError(f"Unknown fusion: {point['fusion']}")

        history = train_validate(train, val, test, model, point,
                                 predictions_dir=PREDICTIONS_DIR)

        # Append this fold pair's rows to the config's CSV
        log_file = build_log_filename(point)
        first_write = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        with open(log_file, "a") as f:
            if first_write:
                f.write(header)
            for h in history:
                f.write(
                    f"{point['fusion']},{point['dataset']},{point['test_set']},{point['dev_set']},"
                    f"{point['arch']},{point['use_pretrained']},{point['augmentation']},"
                    f"{point['learning_rate']},{point['weight_decay']},{effective_batch_size},"
                    f"{h['epoch']},{h['train_loss']:.4f},"
                    f"{h['dev_loss_patient']:.4f},{h['dev_loss_cough']:.4f},"
                    f"{h['dev_acc']:.4f},{h['dev_auc']:.4f},{h['dev_sens']:.4f},{h['dev_spec']:.4f},"
                    f"{h['test_loss_patient']:.4f},{h['test_loss_cough']:.4f},"
                    f"{h['test_acc']:.4f},{h['test_auc']:.4f},{h['test_sens']:.4f},{h['test_spec']:.4f},"
                    f"{h['is_best']}\n"
                )

        completed.add(point_key(point))


if __name__ == "__main__":
    main(grid)
