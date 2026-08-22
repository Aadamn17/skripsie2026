import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# Helper to prepare image batches (list of tensors -> single tensor)
# ======================================================================
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


# ======================================================================
# Self‑Attention Pooling (single head)
# Learnable query vector selects the most important features
# ======================================================================
class SelfAttentionPool(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # Trainable query vector – learns "what a good feature looks like"
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, features):
        # features: (N, dim) – N = number of images (coughs/speeches/pairs)
        # 1. Compute alignment scores (dot product with query)
        scores = torch.matmul(features, self.query)   # (N,)
        # 2. Scale to avoid softmax saturation
        scores = scores / (self.query.size(0) ** 0.5)
        # 3. Convert to weights (sum to 1)
        weights = torch.softmax(scores, dim=0)        # (N,)
        # 4. Weighted sum of features
        pooled = torch.sum(features * weights.unsqueeze(1), dim=0)  # (dim,)
        return pooled


# ======================================================================
# 1. Single‑Modality Classifiers (Cough‑only or Speech‑only)
# ======================================================================
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = resnet18(weights=None)  # No pre‑trained weights
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.resnet(x)


# ======================================================================
# 2. Intermediate Fusion (Separate ResNets + attention pooling)
# ======================================================================
class PatientIntermediateClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        resnet = resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Separate attention pooling per modality
        self.cough_attn = SelfAttentionPool(dim=512)
        self.speech_attn = SelfAttentionPool(dim=512)

        # Optional: unfreeze layer4 if freeze_backbone=False
        if not freeze_backbone:
            for name, param in resnet.named_parameters():
                if 'layer4' in name:
                    param.requires_grad = True

        # Classifier: Logistic Regression on concatenated (1024)
        self.classifier = nn.Linear(1024, num_classes)

    def _encode_modality(self, images, attn_pooler):
        """Encode a list of images and apply attention pooling."""
        device = next(self.parameters()).device
        feats_list = []
        for img in (images or []):
            img = img.to(device=device, dtype=torch.float32)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            feat = self.pool(self.backbone(img)).flatten(1)  # (1,512)
            feats_list.append(feat)
        if feats_list:
            feats = torch.cat(feats_list, dim=0)  # (N,512)
            pooled = attn_pooler(feats)            # (512,)
        else:
            pooled = torch.zeros(512, device=device)
        return pooled

    def _forward_single(self, cough_images, speech_images):
        c_avg = self._encode_modality(cough_images, self.cough_attn)
        s_avg = self._encode_modality(speech_images, self.speech_attn)
        fused = torch.cat([c_avg, s_avg])  # (1024,)
        return self.classifier(fused.unsqueeze(0))  # (1,num_classes)

    def forward(self, cough_images, speech_images):
        if isinstance(cough_images, (list, tuple)) and len(cough_images) > 0 and isinstance(cough_images[0], (list, tuple)):
            logits = []
            for c_imgs, s_imgs in zip(cough_images, speech_images):
                logits.append(self._forward_single(c_imgs, s_imgs))
            return torch.cat(logits, dim=0)
        return self._forward_single(cough_images, speech_images)


# ======================================================================
# 3. True Early Fusion (One ResNet on stacked images + attention pooling)
# ======================================================================
class PatientEarlyImageClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        resnet = resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            for name, param in resnet.named_parameters():
                param.requires_grad = 'layer4' in name

        # Self‑Attention Pooling replaces mean pooling
        self.attention_pool = SelfAttentionPool(dim=512)

        # Classifier: Logistic Regression (Linear 512→2)
        self.classifier = nn.Linear(512, num_classes)

    def _forward_single(self, fused_images):
        device = next(self.parameters()).device
        batch = _prepare_image_batch(fused_images, device)

        if batch.numel() > 0:
            feats = self.pool(self.backbone(batch)).flatten(1)  # (N,512)
            pooled = self.attention_pool(feats)                  # (512,)
        else:
            pooled = torch.zeros(512, device=device)

        return self.classifier(pooled.unsqueeze(0))  # (1,num_classes)

    def forward(self, fused_images):
        if isinstance(fused_images, (list, tuple)) and len(fused_images) > 0 and isinstance(fused_images[0], (list, tuple)):
            logits = []
            for patient_images in fused_images:
                logits.append(self._forward_single(patient_images))
            return torch.cat(logits, dim=0)
        return self._forward_single(fused_images)


# ======================================================================
# 4. Training and Evaluation Functions
# ======================================================================
def _flatten_labels(batch):
    if isinstance(batch, (list, tuple)):
        label_tensor = batch[-1]
    else:
        label_tensor = batch

    if isinstance(label_tensor, torch.Tensor):
        return label_tensor.detach().reshape(-1).cpu()
    if isinstance(label_tensor, (list, tuple)):
        return torch.tensor(label_tensor, dtype=torch.long).reshape(-1)
    return torch.tensor([label_tensor], dtype=torch.long).reshape(-1)


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

            probs = torch.nn.functional.softmax(output, dim=1)[:, 1].detach().cpu()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().reshape(-1).tolist())
    return torch.tensor(all_probs), torch.tensor(all_labels)


def train_validate(train_data, dev_data, test_data, model, params):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["learning_rate"],
                                  weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    # Label smoothing prevents overconfidence
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    dev_acc, dev_auc, test_acc, test_auc = 0.0, 0.0, 0.0, 0.0

    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)

        # Get predictions on dev and test
        dev_probs, dev_labels = get_probs_and_labels(dev_data, model, criterion) if dev_data else (None, None)
        test_probs, test_labels = get_probs_and_labels(test_data, model, criterion) if test_data else (None, None)

        # --- Select threshold on DEV set (not train) ---
        best_thresh = 0.5
        best_f1 = 0.0
        if dev_probs is not None and len(torch.unique(dev_labels)) == 2:
            for thresh in np.arange(0.1, 0.9, 0.02):
                preds = (dev_probs > thresh).float()
                f1 = metrics.f1_score(dev_labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

        # Apply threshold to dev and test
        if dev_probs is not None:
            dev_preds = (dev_probs > best_thresh).float()
            dev_acc = (dev_preds == dev_labels).float().mean().item()
            dev_auc = metrics.roc_auc_score(dev_labels, dev_probs) if len(torch.unique(dev_labels)) == 2 else 0.5

        if test_probs is not None:
            test_preds = (test_probs > best_thresh).float()
            test_acc = (test_preds == test_labels).float().mean().item()
            test_auc = metrics.roc_auc_score(test_labels, test_probs) if len(torch.unique(test_labels)) == 2 else 0.5

        # Use train_loss for the scheduler (no early stopping)
        scheduler.step(train_loss)

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