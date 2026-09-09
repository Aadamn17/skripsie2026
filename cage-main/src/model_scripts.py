# imports
import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logistic regression class
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

# ResNet18 class ->Trying wth frozen backbone and pretrained weights
class ResNet18(nn.Module):
    def __init__(self, num_classes=2,weights = ResNet18_Weights.IMAGENET1K_V1):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(512, num_classes)

        for parmams in self.resnet.parameters():
            parmams.requires_grad = True # True unfreezes the backbone and allows it to be trained. False freezes the backbone and only trains the final layer
    def forward(self, x):
        return self.resnet(x)

class ResNet18_encoder(nn.Module):
    def __init__(self, num_classes=2,weights = ResNet18_Weights.IMAGENET1K_V1):
        super(ResNet18_encoder, self).__init__()
        self.resnet = resnet18(weights=weights)
        #self.resnet.fc = nn.Linear(512, num_classes)
        for parmams in self.resnet.parameters():
            parmams.requires_grad = True # True unfreezes the backbone and allows it to be trained. False freezes the backbone and only trains the final layer
    def forward(self, x):
        return self.resnet(x)
#LateFusion
class LateFusion(nn.Module):
    def __init__(self, speech_encoding_method, num_classes=2, dropout_p=0.3):
        super(LateFusion, self).__init__()
        self.speech_encoding_method = speech_encoding_method
        
        # Encoders
        self.cough_model = ResNet18_encoder(num_classes=num_classes)
        self.cough_model.resnet.fc = nn.Identity()
        
        if self.speech_encoding_method == "lr":
            self.speech_model = Logistic_Regression(input_dim=224*224, num_classes=num_classes)
            in_features = 512 + num_classes
        elif self.speech_encoding_method == "resnet":
            self.speech_model = ResNet18_encoder(num_classes=num_classes)
            self.speech_model.resnet.fc = nn.Identity()
            in_features = 512 + 512

        # 1. Asymmetric Branch Dropout (forces primary dependence on cough)
        self.cough_dropout = nn.Dropout(p=0.2)
        self.speech_dropout = nn.Dropout(p=0.5)

        # 2. Multi-Layer Bottleneck Head with Normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, cough_input, speech_input):
        cough_output = self.cough_model(cough_input)
        speech_output = self.speech_model(speech_input)
        
        # Apply stream-level regularization before concatenation
        cough_output = self.cough_dropout(cough_output)
        speech_output = self.speech_dropout(speech_output)
        
        combined = torch.cat((cough_output, speech_output), dim=1)
        return self.classifier(combined) 

def train_validate(train_data, dev_data, test_data, model, params):
    """
    Training and evaluating logic.
    Returns the development and test accuracies and AUCs from the LAST epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                  weight_decay=params['weight_decay'])
    train_labels = []

    for batch in train_data:
        # Current early-fusion batches are (inputs, labels, patient_ids)
        if len(batch) == 4:  # late fusion: input1, input2, labels, patient_ids
            labels = batch[2]
        else:
            labels = batch[1] # early fusion: inputs, labels, patient_ids

        train_labels.append(labels)

    train_labels = torch.cat(train_labels).long()
    class_counts = torch.bincount(train_labels, minlength=2)

    weights = class_counts.sum() / (2 * class_counts.float())
    weights = weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights) #applying class weights to the loss funciton.

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
    model.train()
    cumulative_loss, total_samples = 0, 0

    for _, batch in enumerate(train_data):
        optimizer.zero_grad()
        
        # Check if late fusion (4 items) or single stream (2 or 3 items)
        if len(batch) == 4:
            cough_data, speech_data, labels, _ = batch
            cough_data = cough_data.to(torch.float32).to(device)
            speech_data = speech_data.to(torch.float32).to(device)
            labels = labels.to(device)
            output = model(cough_data, speech_data)
            batch_size = cough_data.size(0)
        else:
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input_data = input_data.to(torch.float32).to(device)
            labels = labels.to(device)
            output = model(input_data)
            batch_size = input_data.size(0)

        loss = criterion(output, labels.long())
        cumulative_loss += loss.item() * batch_size
        total_samples += batch_size

        loss.backward()
        optimizer.step()

    return cumulative_loss / total_samples

def evaluate_epoch(dev_data, model, criterion):
    """
    Evaluation with patient-level aggregation.
    Supports single-stream inputs (2 or 3 tuple items) and late fusion streams (4 tuple items).
    """
    model.eval()
    cumulative_loss, total_samples = 0, 0
    patient_probs = {}
    patient_labels = {}

    with torch.no_grad():
        for _, batch in enumerate(dev_data):
            # 1. Unpack batch based on modality stream count
            if len(batch) == 4:  # Late fusion: (cough_data, speech_data, labels, pids)
                cough_data, speech_data, labels, pids = batch
                cough_data = cough_data.to(torch.float32).to(device)
                speech_data = speech_data.to(torch.float32).to(device)
                labels = labels.to(device)
                output = model(cough_data, speech_data)
                batch_size = cough_data.size(0)

            elif len(batch) == 3:  # Early fusion / Single-stream with PIDs: (input_data, labels, pids)
                input_data, labels, pids = batch
                input_data = input_data.to(torch.float32).to(device)
                labels = labels.to(device)
                output = model(input_data)
                batch_size = input_data.size(0)

            else:  # Fallback: (input_data, labels)
                input_data, labels = batch
                pids = None
                input_data = input_data.to(torch.float32).to(device)
                labels = labels.to(device)
                output = model(input_data)
                batch_size = input_data.size(0)

            # 2. Extract positive class probabilities (FIXED LINE HERE)
            prob = torch.nn.functional.softmax(output, dim=1)[:, 1]

            # 3. Patient-level aggregation tracking
            if pids is not None:
                for i, pid in enumerate(pids):
                    if pid not in patient_probs:
                        patient_probs[pid] = []
                        patient_labels[pid] = labels[i].item()
                    patient_probs[pid].append(prob[i].item())
            else:
                for i in range(len(labels)):
                    sample_key = f"sample_{total_samples + i}"
                    patient_probs[sample_key] = [prob[i].item()]
                    patient_labels[sample_key] = labels[i].item()

            loss = criterion(output, labels.long())
            cumulative_loss += loss.item() * batch_size
            total_samples += batch_size

    # 4. Average prediction probabilities across patient recordings
    agg_probs = []
    agg_labels = []
    for pid, probs_list in patient_probs.items():
        agg_probs.append(np.mean(probs_list))
        agg_labels.append(patient_labels[pid])

    # 5. Compute performance metrics
    agg_probs = torch.tensor(agg_probs)
    agg_labels = torch.tensor(agg_labels)
    predictions = (agg_probs > 0.5).float()
    
    acc = (predictions == agg_labels).float().mean()
    fpr, tpr, _ = metrics.roc_curve(agg_labels, agg_probs)
    auc = metrics.auc(fpr, tpr)

    cumulative_loss = cumulative_loss / total_samples
    print("Predicted class counts:", torch.bincount(predictions.long(), minlength=2))
    print("Actual class counts:", torch.bincount(agg_labels.long(), minlength=2))
    print("Accuracy:", acc.item())
    
    return cumulative_loss, acc.item(), auc