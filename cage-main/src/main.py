# imports
import itertools # type: ignore
import random, torch # type: ignore
from dataloader import *
from model_scripts import *
  
# Grid for hyperparameter search with 10 fold cross-validation
grid = {
    'loss_selected': ["cross_entropy_resnet"], # Loss functions: "cross_entropy" for logistic regression or "cross_entropy_resnet" for ResNet
    'test_set': [1], 
    'dev_set': [0],
    'num_epochs': [32],
    'batch_size': [32],
    'learning_rate': [1e-4],
    'weight_decay': [1e-4],
    'dataset': ["cage"], # Datasets: "hyfe" or "cage"
    'arch': ["resnet"], # Architectures: "lr" or "resnet"
    'fusion': ["early"], # Fusion strategies: "early" or "intermediate" or "late" or "none"
}

# Log file to keep track of losses and accuracies
log_file = "logs/log.txt"

def main(grid):
    """
    Main function to run the hyperparameter search for logistic regression or Resnet models.
    
    grid: hyperparameter search space for hyperparameter optimisation
    
    Returns the development and test accuracy and AUC
    """
    num_classes = 2 
    bins = 128
    
    # Set seeds for randomness
    random.seed(42)
    torch.manual_seed(42)
    
    # Set device as cude to use the GPU instead of the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(log_file, "a") as file: file.write(f"Architecture, dataset, bins, test_set, dev_set, learning_rate, weight_decay, batch_size, ACC, dev_acc, test_acc, AUC, dev_auc, test_auc\n")
    
    # Loop through all combinations of hyperparameters
    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        params = {**point}
        
        # choose which images to use as input images
        image_type_features = "mel_spectrograms_128" #->for coughs
        
        # Exclude the same experiments for the test and dev set
        if not params['test_set'] == params['dev_set']:
            
            if params['fusion'] == "none":
                train_data, val_data, test_data = get_cough_data(
                    dataset=params['dataset'],
                    data_folds="data/"+params['dataset']+"/data_folds_filtered",
                    i=params['test_set'],
                    j=params['dev_set'],
                    cough_dir="data/"+params['dataset']+"/"+image_type_features,
                    loss=params['loss_selected'],
                    batch_size=params['batch_size'],
                    num_outer_folds=10
                )
            elif params['fusion'] == "early":
                train_data, val_data, test_data = get_early_fusion_data(
                    dataset=params['dataset'],
                    data_folds="data/"+params['dataset']+"/data_folds_filtered",
                    i=params['test_set'],
                    j=params['dev_set'],
                    cough_dir="data/"+params['dataset']+"/"+image_type_features,
                    speech_dir="data/"+params['dataset']+"/mel_spectrograms_counting_128",
                    loss=params['loss_selected'],
                    batch_size=params['batch_size'],
                    num_outer_folds=10
                )
            if params['loss_selected']== "cross_entropy": model = Logistic_Regression(bins, num_classes).to(device) 
            elif params['loss_selected']== "cross_entropy_resnet": model = ResNet18(num_classes=num_classes).resnet.to(device)
            
            dev_acc, dev_auc, test_acc, test_auc = train_validate(train_data, val_data, test_data, model, params)
                
            # Log results
            with open(log_file, "a") as file: file.write(f"{params['arch']}, {params['dataset']}, {128}, {params['test_set']}, {params['dev_set']}, {params['learning_rate']}, {params['weight_decay']}, {params['batch_size']}, ACC, {dev_acc:.6f}, {test_acc:.6f}, AUC, {dev_auc:.6f}, {test_auc:.6f}\n")

    return dev_acc, dev_auc, test_acc, test_auc

if __name__ == "__main__":
    main(grid)