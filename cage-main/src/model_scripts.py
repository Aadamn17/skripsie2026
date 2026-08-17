import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_image_batch(images, device):
    if images is None:
        return torch.empty((0, 3, 224, 224), device=device, dtype=torch.float32)

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
        t = img.to(device=device, dtype=torch.float32)
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
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.resnet(x)


# ------------------------------------------------------------------
# 2. INTERMEDIATE FUSION (Separate ResNets + avg + concat + MLP)
# ------------------------------------------------------------------
class PatientIntermediateClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 🔧 Selective fine‑tuning: only train layer4 + classifier
        if not freeze_backbone:
            for name, param in resnet.named_parameters():
                if 'layer4' in name:   # Unfreeze only the last residual block
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 16),   # reduced complexity
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes)
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
# 3. TRUE EARLY FUSION (One ResNet on stacked images)
# ------------------------------------------------------------------
class PatientEarlyImageClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        if not freeze_backbone:
            for name, param in resnet.named_parameters():
                param.requires_grad = 'layer4' in name   # freeze everything except layer4
        else:
            for param in resnet.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes)
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
# 4. Training and Evaluation Functions (with dynamic threshold)
# ------------------------------------------------------------------
def get_probs_and_labels(loader, model, criterion):
    """Helper: Returns raw probabilities and labels without applying a threshold."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(model, PatientIntermediateClassifier):
                c_imgs, s_imgs, labels = batch
                labels = labels.to(device)
                output = model(c_imgs, s_imgs)
            elif isinstance(model, PatientEarlyImageClassifier):
                f_imgs, labels = batch
                labels = labels.to(device)
                output = model(f_imgs)
            else:
                input_data, labels = batch
                labels = labels.to(device)
                input_data = input_data.to(torch.float32).to(device)
                output = model(input_data)
            prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
            all_probs.append(prob)
            all_labels.append(labels.item())
    return torch.tensor(all_probs), torch.tensor(all_labels)


def train_validate(train_data, dev_data, test_data, model, params):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["learning_rate"],
                                  weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # --- compute class weights from the training set ---
    all_train_labels = []
    for batch in train_data:
        all_train_labels.append(batch[-1].item())
    all_train_labels = torch.tensor(all_train_labels)
    class_counts = torch.bincount(all_train_labels.long(), minlength=2)
    class_weights = (1.0 / class_counts.float())
    class_weights = (class_weights / class_weights.sum() * len(class_counts)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    dev_acc, dev_auc, test_acc, test_auc = 0.0, 0.0, 0.0, 0.0

    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)

        # Get raw probabilities and labels for Train, Dev and Test
        train_probs, train_labels = get_probs_and_labels(train_data, model, criterion)
        dev_probs, dev_labels = get_probs_and_labels(dev_data, model, criterion) if dev_data else (None, None)
        test_probs, test_labels = get_probs_and_labels(test_data, model, criterion) if test_data else (None, None)

        # --- THRESHOLD SELECTION ON TRAIN, NOT DEV ---
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.02):
            preds = (train_probs > thresh).float()
            f1 = metrics.f1_score(train_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # Apply that fixed threshold to Dev and Test, no further searching
        if dev_probs is not None:
            dev_preds = (dev_probs > best_thresh).float()
            dev_acc = (dev_preds == dev_labels).float().mean().item()
            dev_auc = metrics.roc_auc_score(dev_labels, dev_probs) if len(torch.unique(dev_labels)) == 2 else 0.5

        if test_probs is not None:
            test_preds = (test_probs > best_thresh).float()
            test_acc = (test_preds == test_labels).float().mean().item()
            test_auc = metrics.roc_auc_score(test_labels, test_probs) if len(torch.unique(test_labels)) == 2 else 0.5

        with open("logs/per_epoch_loss.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{params['num_epochs']}, Train Loss: {train_loss:.4f}, "
                    f"Dev Acc: {dev_acc:.4f}, Dev AUC: {dev_auc:.4f}, "
                    f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, "
                    f"Best Thresh: {best_thresh:.2f}\n")
        print(f"[epoch {epoch+1}] train_loss={train_loss:.4f}, "
              f"dev_auc={dev_auc:.4f}, best_thresh={best_thresh:.2f}")

    return dev_acc, dev_auc, test_acc, test_auc


def train_epoch(train_data, model, optimizer, criterion):
    model.train()
    total_loss, num_samples = 0.0, 0
    for batch in train_data:
        if isinstance(model, PatientIntermediateClassifier):
            c_imgs, s_imgs, labels = batch
            labels = labels.to(device)
            output = model(c_imgs, s_imgs)
        elif isinstance(model, PatientEarlyImageClassifier):
            f_imgs, labels = batch
            labels = labels.to(device)
            output = model(f_imgs)
        else:
            input_data, labels = batch
            labels = labels.to(device)
            input_data = input_data.to(torch.float32).to(device)
            output = model(input_data)

        loss = criterion(output, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_samples += labels.size(0)
    return total_loss / num_samples