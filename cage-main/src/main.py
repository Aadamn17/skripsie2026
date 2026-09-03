import itertools
import random
import torch
from dataloader import *
from model_scripts import train_validate, ResNet18

grid = {
    'loss_selected': ["cross_entropy_resnet"],
    'test_set': [0,1,2,3,4,5,6,7,8,9],
    'dev_set': [0,1,2,3,4,5,6,7,8,9],
    'num_epochs': [32],
    'batch_size': [32],
    'learning_rate': [1e-4],
    'weight_decay': [1e-4],
    'dataset': ["cage"],
    'arch': ["resnet"],
    'fusion': ["early"],   # "early" or "late" or "none"
    'augmentation': ["gaussian_noise"] #"gaussian_noise", "solarisation", "frequency_masking", "time_masking"
}

cough_dir = "data/cage/mel_spectrograms_128"
speech_dir = "data/cage/mel_spectrograms_counting_128"
log_file = "logs/log.txt"

def main(grid):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_file, "a") as f:
        f.write("fusion,dataset,test_set,dev_set,augmentation,lr,wd,batch_size,dev_acc,dev_auc,test_acc,test_auc\n")

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        if point['test_set'] == point['dev_set']:
            continue

        if point['fusion'] == "early":
            train, val, test = get_early_fusion_data(
                dataset=point['dataset'],
                data_folds="data/"+point['dataset']+"/data_folds_filtered",
                i=point['test_set'], j=point['dev_set'],
                cough_dir=cough_dir, speech_dir=speech_dir,
                loss=point['loss_selected'], batch_size=point['batch_size'],
                num_outer_folds=10,
                augmentation=point['augmentation']
            )
            model = ResNet18(num_classes=2).to(device)

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

        elif ['fusion'] == 'late':
            train, val, test = get_late_fusion_data(
                dataset=point['cage'],
                data_folds = "data/"+point['dataset']+"data_folds_filtered",
                i=point['test_set'],j=point['dev_set'],
                cough_dir=cough_dir,speech_dir=speech_dir,
                loss = point['loss_selected'],batch_size = point['batch_size'],
                num_outer_folds=10,
                augmentation = point['augmentation']

            )
            #model = LateFusion

        else:
            raise ValueError(f"Unknown fusion: {point['fusion']}")

        dev_acc, dev_auc, test_acc, test_auc = train_validate(train, val, test, model, point)

        with open(log_file, "a") as f:
            f.write(f"{point['fusion']},{point['dataset']},{point['test_set']},{point['dev_set']},"
                    f"{point['augmentation']},{point['learning_rate']},{point['weight_decay']},{point['batch_size']},"
                    f"{dev_acc:.4f},{dev_auc:.4f},{test_acc:.4f},{test_auc:.4f}\n")

if __name__ == "__main__":
    main(grid)