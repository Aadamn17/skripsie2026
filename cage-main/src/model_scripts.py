# imports
import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logistic regression class
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

# ResNet18 class (original)
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.resnet(x)

#LateFusion
class LateFusion(nn.Module):
    def __init__(self,num_classes=2):
        super(LateFusion,self).__init__()
        self.cough_model = ResNet18(num_classes=num_classes)
        self.speech_model = Logistic_Regression(input_dim=128,num_classes=num_classes)
        self.fc = nn.Linear(2*num_classes,num_classes)
    def forward(self,cough_input,speech_input):
        cough_output = self.cough_model(cough_input)
        speech_output = self.speech_model(speech_input)
        combined = torch.cat((cough_output,speech_output),dim=1)
        return self.fc(combined)    

def train_validate(train_data, dev_data, test_data, model, params):
    """
    Training and evaluating logic.
    Returns the development and test accuracies and AUCs from the LAST epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                  weight_decay=params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    dev_acc, dev_auc, test_acc, test_auc = 0, 0, 0, 0
    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)
        if dev_data is not None:
            dev_loss, dev_acc, dev_auc = evaluate_epoch(dev_data, model, criterion)
        if test_data is not None:
            test_loss, test_acc, test_auc = evaluate_epoch(test_data, model, criterion)
        with open("logs/per_epoch_loss.txt", "a") as file:
            file.write(f"Epoch {epoch+1}/{params['num_epochs']}, Train Loss: {train_loss:.4f}, "
                       f"Dev Loss: {dev_loss:.4f}, Test Loss: {test_loss:.4f}, "
                       f"Dev Acc: {dev_acc:.4f}, Dev AUC: {dev_auc:.4f}, "
                       f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}\n")
    return dev_acc, dev_auc, test_acc, test_auc

def train_epoch(train_data, model, optimizer, criterion):
    """
    Training loop. Each batch is (input_data, labels, _) or (input_data, labels) depending on the dataloader.
    Loss is computed exactly as the original: criterion(output[:,1], labels.float()).
    """
    model.train()
    cumulative_loss, total_samples = 0, 0

    for _, input in enumerate(train_data):
        optimizer.zero_grad()
        # Handle both 2-item and 3-item tuple (if patient ID is present, ignore it)
        if len(input) == 3:
            input_data, labels, _ = input
        else:
            input_data, labels = input
        labels = labels.to(device)
        input_data = input_data.to(torch.float32).to(device)
        output = model(input_data).to(device)

        # Original loss calculation 
        loss = criterion(output[:,1], labels.to(torch.float))

        cumulative_loss += loss.item()
        total_samples += input_data.size(0)

        loss.backward()
        optimizer.step()

    cumulative_loss = cumulative_loss / total_samples
    return cumulative_loss

def evaluate_epoch(dev_data, model, criterion):
    """
    Evaluation with patient-level aggregation.
    Each batch is (input_data, labels, pids) or (input_data, labels).
    Loss is computed exactly as the original: criterion(output[:,1], labels.float()).
    """
    model.eval()
    cumulative_loss, total_samples = 0, 0
    patient_probs = {}
    patient_labels = {}

    with torch.no_grad():
        for _, input in enumerate(dev_data):
            if len(input) == 3:
                input_data, labels, pids = input
            else:
                input_data, labels = input
                pids = None  # No patient IDs available (use cough-level)

            input_data = input_data.to(torch.float32).to(device)
            output = model(input_data).to(device)
            labels = labels.to(device)

            prob = torch.nn.functional.softmax(output, dim=1)[:, 1]  # p(positive)

            if pids is not None:
                # Patient-level aggregation
                for i, pid in enumerate(pids):
                    if pid not in patient_probs:
                        patient_probs[pid] = []
                        patient_labels[pid] = labels[i].item()
                    patient_probs[pid].append(prob[i].item())
            else:
                # Fallback to cough-level if no patient IDs
                for i in range(len(labels)):
                    if str(i) not in patient_probs:
                        patient_probs[str(i)] = []
                        patient_labels[str(i)] = labels[i].item()
                    patient_probs[str(i)].append(prob[i].item())
            class_counts = torch.bincount(labels)
            class_weights = 1.0/class_counts.float()
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights) # apply class weighting to reduce overconfidence
            loss = criterion(output, labels)   # labels are LongTensor: [0, 1, 0, 1, ...]
            cumulative_loss += loss.item()
            total_samples += input_data.size(0)

    # Aggregate per patient (or per sample if no patient IDs)
    agg_probs = []
    agg_labels = []
    for pid, probs_list in patient_probs.items():
        agg_probs.append(np.mean(probs_list))
        agg_labels.append(patient_labels[pid])

    # Compute metrics
    agg_probs = torch.tensor(agg_probs)
    agg_labels = torch.tensor(agg_labels)
    predictions = (agg_probs > 0.5).float()
    acc = (predictions == agg_labels).float().mean()
    fpr, tpr, _ = metrics.roc_curve(agg_labels, agg_probs)
    auc = metrics.auc(fpr, tpr)

    cumulative_loss = cumulative_loss / total_samples
    return cumulative_loss, acc, auc