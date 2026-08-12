# imports
import torch # type: ignore
import numpy as np # type: ignore
from sklearn import metrics # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision.models import resnet18 #type: ignore

# Set device as cude to use the GPU instead of the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logistic regression class
class Logistic_Regression(nn.Module):
    """
    Logistic regression model. 
    Note that a softmax or sigmoid activation is not applied since this model is used with CrossEntropyLoss which expects the logits.
    
    input_dim: number of frequency bins
    num_classes: number of classes
    
    Returns the logits (batch size, 2)
    """
    def __init__(self, input_dim=128, num_classes=2):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.linear(x)
        return x
    
class ResNet18(nn.Module):
    """
    ResNet18 model from Pytorch. Last layer is replaced with a linear layer that gives num_classes outputs instead of a 1000.
    
    Returns the output probabilities (batch size, 2)
    """
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.resnet = resnet18() 
        self.resnet.fc = nn.Linear(512, num_classes) 

class EarlyFuser(nn.Module):
    '''input dimensions = num freq bins
    Example usage: fuser = TensorFuser(tensors=[sensor_a_features, sensor_b_features])'''

    def __init__(self,input_dim=128,num_classes=2):
        super(EarlyFuser,self).___init__()
        self.early = EarlyFuser
def train_validate(train_data, dev_data, test_data, model, params):
    """
    Training and evaluating logic for the model
    
    train_data: DataLoader object that contains the training data
    dev_data: DataLoader object that contains the development data
    test_data: DataLoader object that contains the test data
    model: LR or ResNet18 model
    params: parameters selected in current iteration of the grid optimisation
    
    Returns the development and test accuracies and AUCs
    """
    # AdamW or Adam can be used for the optimizer
    # Note that since the Pytorch CrossEntropyLoss is used, the criterion expects the logits not the probabilities, so the LR model does not have a softmax/sigmoid layer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params['weight_decay'])                                    
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model and evaluate it on the dev set or the test set for a set number of epochs
    dev_acc, dev_auc, test_acc, test_auc = 0,0,0,0
    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)
        if not dev_data is None: dev_loss, dev_acc, dev_auc = evaluate_epoch(dev_data, model, criterion)
        if not test_data is None: test_loss, test_acc, test_auc = evaluate_epoch(test_data, model, criterion)
        with open("logs/per_epoch_loss.txt", "a") as file: file.write(f"Epoch {epoch+1}/{params['num_epochs']}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Test Loss: {test_loss:.4f}, Dev Acc: {dev_acc:.4f}, Dev AUC: {dev_auc:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}\n")

    return dev_acc, dev_auc, test_acc, test_auc

def train_epoch(train_data, model, optimizer, criterion):
    """
    Training logic
    
    train_data: DataLoader object that contains the training data
    model: LR or ResNet18 model
    optimizer: AdamW optimizer
    criterion: CrossEntroyLoss loss function
    
    Returns the cummulative loss for all the batches
    """
    model.train()
    cumulative_loss, total_samples, loss = 0, 0,0
    
    for _, input in enumerate(train_data):
        optimizer.zero_grad()    
        input_data, labels = input
        labels = labels.to(device)
        input_data = input_data.to(torch.float32).to(device)
        output = model(input_data).to(device)
        loss = criterion(output[:,1], labels.to(torch.float))
        cumulative_loss += loss.item()
        total_samples += input_data.size(0)
        
        loss.backward()
        optimizer.step()
        
    cumulative_loss = cumulative_loss/total_samples
    return cumulative_loss

def evaluate_epoch(dev_data, model, criterion):
    """
    Evaluating logic
    
    dev_data: DataLoader object that contains the development or test data
    model: LR or ResNet18 model
    criterion: CrossEntroyLoss loss function
    
    """
    model.eval()
    cumulative_loss, acc, auc, total_samples = 0, 0, 0, 0
    outputs = []
    labels = []
    probs = []
    
    with torch.no_grad():
        for steps, input in enumerate(dev_data):
            input_data, label = input
            input_data = input_data.to(torch.float32).to(device)
            output = model(input_data).to(device)
            label = label.to(device).to(torch.float)
            prob = torch.nn.functional.softmax(output, dim=1)
            labels.extend(label.detach().cpu().numpy())
            probs.extend(prob[:,1].detach().cpu().numpy())
            loss = criterion(output[:,1], label)
            cumulative_loss += loss.item()
            total_samples += input_data.size(0)

    labels = torch.tensor(labels)
    probs = torch.tensor(probs)
    
    predictions = (probs > 0.5).float()
    acc = (predictions == labels).float().mean()
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc = metrics.auc(fpr, tpr)
    
    cumulative_loss = cumulative_loss/total_samples

    return cumulative_loss, acc, auc

def average_pool(input_tensor):
    "Calculated average per filter bank of a spectogram"
    pooled_tensor = np.mean(input_tensor,axis=1)

    return pooled_tensor
