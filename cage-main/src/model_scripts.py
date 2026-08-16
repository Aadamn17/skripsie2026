import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_image_batch(images, device):
    if images is None:
        return torch.empty((0, 3, 224, 224), device=device,dtype=torch.float32)

    if isinstance(images, torch.Tensor):
        x = images.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected a 3D/4D/5D image tensor, got shape {tuple(x.shape)}")
        return x

    batches = []
    for img in images:
        t = img.to(device = device, dtype = torch.float32)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        elif t.dim() == 5 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.dim() != 4:
            raise ValueError(f"Expected each image to be [C, H, W] or [N, C, H, W], got {tuple(t.shape)}")
        batches.append(t)

    if not batches:
        return torch.empty((0, 3, 224, 224), device=device)

    return torch.cat(batches, dim=0)

# ------------------------------------------------------------------
# 1. Single-Modality Classifiers
# ------------------------------------------------------------------
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.resnet(x)

# ------------------------------------------------------------------
# 2. INTERMEDIATE FUSION (Your described method)
#    Separate ResNets per modality -> average features -> concat -> MLP
# ------------------------------------------------------------------
class PatientIntermediateClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(PatientIntermediateClassifier, self).__init__()
        # Shared feature extractor
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, cough_images, speech_images):
        device = next(self.parameters()).device
        c_batch = _prepare_image_batch(cough_images, device)
        s_batch = _prepare_image_batch(speech_images, device)

        if c_batch.numel() > 0:
            c_feats = self.pool(self.backbone(c_batch)).flatten(1)
            c_avg = c_feats.mean(dim=0, keepdim=True)
        else:
            c_avg = torch.zeros(1, 512, device=device)

        if s_batch.numel() > 0:
            s_feats = self.pool(self.backbone(s_batch)).flatten(1)
            s_avg = s_feats.mean(dim=0, keepdim=True)
        else:
            s_avg = torch.zeros(1, 512, device=device)

        fused = torch.cat([c_avg, s_avg], dim=1)
        return self.classifier(fused)

# ------------------------------------------------------------------
# 3. TRUE EARLY FUSION
#    One ResNet on stacked (Cough, Speech, Avg) images -> average features -> MLP
# ------------------------------------------------------------------
class PatientEarlyImageClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(PatientEarlyImageClassifier, self).__init__()
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, fused_images):
        device = next(self.parameters()).device
        batch = _prepare_image_batch(fused_images, device)

        if batch.numel() > 0:
            feats = self.pool(self.backbone(batch)).flatten(1)
            avg_feat = feats.mean(dim=0, keepdim=True)
        else:
            avg_feat = torch.zeros(1, 512, device=device)
        return self.classifier(avg_feat)

# ------------------------------------------------------------------
# 4. Training and Evaluation Functions
# ------------------------------------------------------------------
def _debug_label_distribution(data_loader, name):
    counts = {0: 0, 1: 0}
    for batch in data_loader:
        labels = batch[-1]
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
    print(f"[{name}] label distribution: {counts}")
    return counts


def train_validate(train_data, dev_data, test_data, model, params):
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params['weight_decay'])

    # --- compute class weights from the training set ---
    all_train_labels = []
    for batch in train_data:
        labels = batch[-1]          # labels is always the last element of the batch tuple
        all_train_labels.append(labels.item())
    all_train_labels = torch.tensor(all_train_labels)

    class_counts = torch.bincount(all_train_labels.long(), minlength=2)   # [count_class0, count_class1]
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize so weights are ~scale 1
    class_weights = class_weights.to(device)

    print(f"[train] raw class counts: {dict(zip([0, 1], class_counts.tolist()))}")
    print(f"[train] effective class weights: {class_weights.detach().cpu().tolist()}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # --- end of change ---

    dev_acc, dev_auc, test_acc, test_auc = 0.0, 0.0, 0.0, 0.0

    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)
        if dev_data is not None:
            dev_loss, dev_acc, dev_auc = evaluate_epoch(dev_data, model, criterion)
        if test_data is not None:
            test_loss, test_acc, test_auc = evaluate_epoch(test_data, model, criterion)
        with open("logs/per_epoch_loss.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{params['num_epochs']}, Train Loss: {train_loss:.4f}, Dev Acc: {dev_acc:.4f}, Dev AUC: {dev_auc:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}\n")
        print(f"[epoch {epoch+1}] train_loss={train_loss:.4f}, dev_acc={dev_acc:.4f}, dev_auc={dev_auc:.4f}")
    return dev_acc, dev_auc, test_acc, test_auc

def train_epoch(train_data, model, optimizer, criterion):
    model.train()
    total_loss, num_samples = 0.0, 0
    for batch in train_data:
        if isinstance(model, PatientIntermediateClassifier):
            c_imgs, s_imgs, labels = batch
            labels = labels.to(device)
            output = model(c_imgs, s_imgs)
            loss = criterion(output, labels.long())
        elif isinstance(model, PatientEarlyImageClassifier):
            f_imgs, labels = batch
            labels = labels.to(device)
            output = model(f_imgs)
            loss = criterion(output, labels.long())
        else:
            input_data, labels = batch
            labels = labels.to(device)
            input_data = input_data.to(torch.float32).to(device)
            output = model(input_data)
            loss = criterion(output, labels.long())

        pred_class = output.argmax(dim=1)
        if num_samples == 0:
            print(f"[train batch sample] labels={labels.detach().cpu().tolist()[:10]}, preds={pred_class.detach().cpu().tolist()[:10]}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_samples += labels.size(0)
    return total_loss / num_samples

def evaluate_epoch(dev_data, model, criterion):
    model.eval()
    total_loss, num_samples = 0.0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dev_data:
            if isinstance(model, PatientIntermediateClassifier):
                c_imgs, s_imgs, labels = batch
                labels = labels.to(device)
                output = model(c_imgs, s_imgs)
                loss = criterion(output, labels.long())
                prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
            elif isinstance(model, PatientEarlyImageClassifier):
                f_imgs, labels = batch
                labels = labels.to(device)
                output = model(f_imgs)
                loss = criterion(output, labels.long())
                prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
            else:
                input_data, labels = batch
                labels = labels.to(device)
                input_data = input_data.to(torch.float32).to(device)
                output = model(input_data)
                loss = criterion(output, labels.long())
                prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()

            total_loss += loss.item()
            all_labels.append(labels.item())
            all_probs.append(prob)
            num_samples += labels.size(0)

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    preds = (all_probs > 0.5).float()
    acc = (preds == all_labels).float().mean().item()
    if len(torch.unique(all_labels)) == 2:
        fpr, tpr, _ = metrics.roc_curve(all_labels.numpy(), all_probs.numpy())
        auc = metrics.auc(fpr, tpr)
    else:
        auc = 0.5
    return total_loss / num_samples, acc, auc