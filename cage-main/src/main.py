import itertools
import random
import torch
from dataloader import *
from model_scripts import *

# ------------------------------------------------------------------
# Hyperparameter grid (supports both fusion methods)
# ------------------------------------------------------------------
grid = {
    'loss_selected': ["cross_entropy_resnet"],
    'test_set': [1], 
    'dev_set': [0],
    'num_epochs': [10],
    'batch_size': [1],        # Batch size 1 for patient-level lists
    'learning_rate': [1e-4],
    'weight_decay': [1e-4],
    'dataset': ["cage"],
    'arch': ["resnet"],
    # 'fusion': ["intermediate_feature", "early_image", "none"]  # Uncomment to run all
    'fusion': ["intermediate_feature"]  # Change this to "intermediate_feature" to run your method
}

log_file = "logs/log.txt"

def main(grid):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_file, "a") as f:
        f.write("fusion,dataset,test_set,dev_set,lr,wd,batch_size,dev_acc,dev_auc,test_acc,test_auc\n")

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        if point['test_set'] == point['dev_set']:
            continue

        image_type_features = "mel_spectrograms_128"
        base_path = "data/" + point['dataset']
        data_folds = base_path + "/data_folds_filtered"
        cough_dir = base_path + "/" + image_type_features
        speech_dir = base_path + "/mel_spectrograms_counting_128"

        # ------------------------------------------------------------------
        # 1. Select Data and Model based on Fusion Type
        # ------------------------------------------------------------------
        if point['fusion'] == "none":
            train, val, test = get_cough_data(
                dataset=point['dataset'], data_folds=data_folds,
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, loss=point['loss_selected'],
                batch_size=point['batch_size'], num_outer_folds=10
            )
            model = ResNet18(num_classes=2).to(device)

        elif point['fusion'] == "intermediate_feature":
            train, val, test = get_intermediate_data(
                dataset=point['dataset'], data_folds=data_folds,
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10
            )
            model = PatientIntermediateClassifier(num_classes=2,freeze_backbone=False).to(device)

        elif point['fusion'] == "early_image":
            train, val, test = get_early_image_data(
                dataset=point['dataset'], data_folds=data_folds,
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10
            )
            model = PatientEarlyImageClassifier(num_classes=2,freeze_backbone=False).to(device)

        else:
            raise ValueError(f"Unknown fusion method: {point['fusion']}")

        # ------------------------------------------------------------------
        # 2. Train and Evaluate
        # ------------------------------------------------------------------
        dev_acc, dev_auc, test_acc, test_auc = train_validate(train, val, test, model, point)

        # ------------------------------------------------------------------
        # 3. Log Results
        # ------------------------------------------------------------------
        with open(log_file, "a") as f:
            f.write(f"{point['fusion']},{point['dataset']},{point['test_set']},{point['dev_set']},"
                    f"{point['learning_rate']},{point['weight_decay']},{point['batch_size']},"
                    f"{dev_acc:.4f},{dev_auc:.4f},{test_acc:.4f},{test_auc:.4f}\n")

if __name__ == "__main__":
    main(grid)