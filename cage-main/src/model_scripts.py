# imports
import copy
import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# EARLY STOPPING
# ======================================================================

class EarlyStopper:
    """Stops training when the monitored metric hasn't improved for `patience`
    epochs. Supports mode='min' (loss) and mode='max' (AUC). Smooths over
    `window` epochs to suppress single-epoch noise, and never stops before
    `min_epochs`."""
    def __init__(self, patience=7, min_delta=1e-3, mode="max",
                 window=3, min_epochs=10):
        self.patience    = patience
        self.min_delta   = min_delta
        self.mode        = mode
        self.window      = window
        self.min_epochs  = min_epochs
        self.best        = float("inf") if mode == "min" else -float("inf")
        self.counter     = 0
        self.best_state  = None
        self.best_epoch  = 0
        self.should_stop = False
        self.history     = []

    def step(self, value, model, epoch):
        # NaN (e.g. AUC undefined when only one class is present) -> non-improvement
        if value is None or (isinstance(value, float) and value != value):
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.should_stop = True
            return False

        self.history.append(value)
        if len(self.history) < self.window:
            return False
        smoothed = sum(self.history[-self.window:]) / self.window

        if self.mode == "min":
            improved = smoothed < self.best - self.min_delta
        else:
            improved = smoothed > self.best + self.min_delta

        if improved:
            self.best       = smoothed
            self.counter    = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.should_stop = True
            return False

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Logistic_Regression(nn.Module):
    def __init__(self, fusion_type, input_dim, num_classes=2):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim,num_classes)

    def forward(self, x):
        return self.linear(x)


class ResNet18(nn.Module):
    """Single-stream ResNet18. fusion_type='late' leaves FC untouched
    (used only when you want the raw encoder head)."""
    def __init__(self, fusion_type="none", num_classes=2, dropout_p=0.3,
                 weights=ResNet18_Weights.IMAGENET1K_V1):
        super(ResNet18, self).__init__()
        self.fusion_type = fusion_type
        self.resnet = resnet18(weights=weights)

        for p in self.resnet.parameters():
            p.requires_grad = False

        if self.fusion_type != "late":
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(512, num_classes),
            )
        else:
            # fc replaced by Dropout — but it will be overwritten by LateFusion
            self.resnet.fc = nn.Dropout(p=dropout_p)

        '''for p in self.resnet.layer3.parameters():
            p.requires_grad = True'''
        for p in self.resnet.layer4.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


class ResNet18_encoder(nn.Module):
    """ResNet18 backbone that outputs a 512-d feature vector (no classifier)."""
    def __init__(self, num_classes=2, dropout_p=0.5,
                 weights=ResNet18_Weights.IMAGENET1K_V1):
        super(ResNet18_encoder, self).__init__()
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Identity()   # -> [B, 512]
        self.dropout = nn.Dropout(p=dropout_p)

        for p in self.resnet.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return self.dropout(x)


class LateFusion(nn.Module):
    def __init__(self, speech_encoding_method, num_classes=2, dropout_p=0.3):
        super(LateFusion, self).__init__()
        self.speech_encoding_method = speech_encoding_method

        # Encoders
        self.cough_model = ResNet18_encoder(num_classes=num_classes)
        # cough_model.resnet.fc is already nn.Identity()

        if self.speech_encoding_method == "lr":
            # LR -> 512 so it can concat with the cough encoder output
            self.speech_model = Logistic_Regression(
                fusion_type="late", input_dim=224 * 224, num_classes=num_classes
            )
            in_features = 512 + 512
        elif self.speech_encoding_method == "resnet":
            self.speech_model = ResNet18_encoder(num_classes=num_classes)
            in_features = 512 + 512
        else:
            raise ValueError(f"Unknown speech_encoding_method: {speech_encoding_method}")

        # Stream-level dropout
        self.cough_dropout  = nn.Dropout(p=0.3)
        self.speech_dropout = nn.Dropout(p=0.3)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, cough_input, speech_input):
        cough_output = self.cough_model(cough_input)

        # LR expects [B, 224*224]; speech_input arrives as [B, 1, 224, 224]
        if self.speech_encoding_method == "lr":
            speech_input = speech_input.view(speech_input.size(0), -1)

        speech_output = self.speech_model(speech_input)

        cough_output  = self.cough_dropout(cough_output)
        speech_output = self.speech_dropout(speech_output)

        combined = torch.cat((cough_output, speech_output), dim=1)
        return self.classifier(combined)


class Intermediate(nn.Module):
    def __init__(self, num_classes):
        super(Intermediate, self).__init__()
        resnetcough = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cough_base = nn.Sequential(
            resnetcough.conv1,
            resnetcough.bn1,
            resnetcough.relu,
            resnetcough.maxpool,
            resnetcough.layer1,
            resnetcough.layer2,
            resnetcough.layer3,
        )
        resnetspeech = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.speech_base = nn.Sequential(
            resnetspeech.conv1,
            resnetspeech.bn1,
            resnetspeech.relu,
            resnetspeech.maxpool,
            resnetspeech.layer1,
            resnetspeech.layer2,
            resnetspeech.layer3,
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4  = resnetcough.layer4
        self.avgpool = resnetcough.avgpool
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, stream1, stream2):
        c_feat = self.cough_base(stream1)
        s_feat = self.speech_base(stream2)
        fused  = torch.cat((c_feat, s_feat), dim=1)
        fused  = self.fusion_conv(fused)
        out    = self.layer4(fused)
        out    = self.avgpool(out)
        out    = torch.flatten(out, 1)
        out    = self.fc(out)
        return out


def train_validate(train_data, dev_data, test_data, model, params):
    """
    Training with early stopping on validation loss.
    Returns per-epoch history list of dicts and restores best weights.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    # ---- class weights (per-cough, from training fold) ----
    train_labels = []
    for batch in train_data:
        labels = batch[2] if len(batch) == 4 else batch[1]
        train_labels.append(labels)
    train_labels = torch.cat(train_labels).long()
    class_counts = torch.bincount(train_labels, minlength=2)
    class_counts = torch.clamp(class_counts, min=1)          # guard absent class
    weights = class_counts.sum() / (2 * class_counts.float())
    weights = weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # ---- early stopping ----
    patience  = params.get("early_stop_patience", 7)
    min_delta = params.get("early_stop_min_delta", 1e-4)
    stopper   = EarlyStopper(patience=patience, min_delta=min_delta, mode="min")

    dev_fold  = params.get("dev_set")
    test_fold = params.get("test_set")

    history = []

    # ---- header (once per fold) ----
    print(
        f"\n{'=' * 120}\n"
        f"Fold: test={test_fold}  dev={dev_fold}  "
        f"fusion={params.get('fusion', '?')}  arch={params.get('arch', '?')}  "
        f"aug={params.get('augmentation', '?')}  patience={patience}\n"
        f"{'-' * 120}\n"
        f"{'ep':>4} | {'tr_loss_cough':>13} | "
        f"{'dev_loss_pat':>12} {'dev_loss_cough':>14} {'dev_acc':>8} {'dev_auc':>8} | "
        f"{'test_loss_pat':>13} {'test_loss_cough':>15} {'test_acc':>9} {'test_auc':>9} | notes\n"
        f"{'-' * 120}"
    )

    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)

        dev_loss_pat, dev_acc, dev_auc, dev_loss_cough = (
            float("inf"), 0.0, float("nan"), float("inf")
        )
        if dev_data is not None:
            dev_loss_pat, dev_acc, dev_auc, dev_loss_cough = evaluate_epoch(
                dev_data, model, criterion, set_name="dev", verbose=False
            )

        test_loss_pat, test_acc, test_auc, test_loss_cough = (
            float("inf"), 0.0, float("nan"), float("inf")
        )
        if test_data is not None:
            test_loss_pat, test_acc, test_auc, test_loss_cough = evaluate_epoch(
                test_data, model, criterion, set_name="test", verbose=False
            )

        # ---- early-stopping step ----
        if dev_data is not None:
            is_best = stopper.step(dev_loss_pat, model, epoch)
        else:
            is_best = stopper.step(train_loss, model, epoch)

        history.append({
            "epoch":             epoch + 1,
            "train_loss":        train_loss,
            "dev_loss_patient":  dev_loss_pat,
            "dev_loss_cough":    dev_loss_cough,
            "dev_acc":           dev_acc,
            "dev_auc":           dev_auc,
            "test_loss_patient": test_loss_pat,
            "test_loss_cough":   test_loss_cough,
            "test_acc":          test_acc,
            "test_auc":          test_auc,
            "is_best":           int(is_best),
        })

        # ---- terminal row ----
        note = ""
        if is_best:
            note += " *best*"
        if stopper.should_stop:
            note += " [STOP]"

        print(
            f"{epoch + 1:>4} | {train_loss:>13.4f} | "
            f"{dev_loss_pat:>12.4f} {dev_loss_cough:>14.4f} "
            f"{dev_acc:>8.4f} {dev_auc:>8.4f} | "
            f"{test_loss_pat:>13.4f} {test_loss_cough:>15.4f} "
            f"{test_acc:>9.4f} {test_auc:>9.4f} |{note}"
        )

        if stopper.should_stop:
            print(
                f"[EARLY STOP] epoch {epoch + 1}: no dev-loss improvement "
                f"for {patience} epochs. Best epoch = {stopper.best_epoch + 1}."
            )
            break

    # ---- restore best weights and print final summary ----
    if dev_data is not None:
        stopper.restore_best(model)
        print(f"\n[FINAL] restored best epoch = {stopper.best_epoch + 1}")
        evaluate_epoch(dev_data, model, criterion, set_name="dev_best", verbose=True)
        if test_data is not None:
            evaluate_epoch(test_data, model, criterion, set_name="test_best", verbose=True)

    return history


def train_epoch(train_data, model, optimizer, criterion):
    model.train()
    cumulative_loss, total_samples = 0, 0

    for _, batch in enumerate(train_data):
        optimizer.zero_grad()

        if len(batch) == 4:
            cough_data, speech_data, labels, _ = batch
            cough_data  = cough_data.to(torch.float32).to(device)
            speech_data = speech_data.to(torch.float32).to(device)
            labels      = labels.to(device)
            output = model(cough_data, speech_data)
            batch_size = cough_data.size(0)
        else:
            if len(batch) == 3:
                input_data, labels, _ = batch
            else:
                input_data, labels = batch
            input_data = input_data.to(torch.float32).to(device)
            labels     = labels.to(device)
            output = model(input_data)
            batch_size = input_data.size(0)

        loss = criterion(output, labels.long())
        cumulative_loss += loss.item() * batch_size
        total_samples   += batch_size

        loss.backward()
        optimizer.step()

    return cumulative_loss / total_samples


def evaluate_epoch(dev_data, model, criterion, set_name="eval", verbose=True):
    """
    Patient-level evaluation.

    Forward pass per cough -> collect logits -> average logits per patient
    (Bayesian pooling) -> compute per-patient loss, accuracy, AUC.

    Also returns the per-cough loss (same definition as in train_epoch)
    so you can spot divergence between cough- and patient-level behavior.

    Supports 2-tuple, 3-tuple, and 4-tuple (late fusion) batches.
    Returns: (patient_loss, acc, auc, cough_loss)
    """
    model.eval()

    cumulative_cough_loss, total_coughs = 0.0, 0
    patient_logits, patient_labels = {}, {}
    sample_counter = 0

    with torch.no_grad():
        for batch in dev_data:
            # ---------- forward pass ----------
            if len(batch) == 4:
                cough_data, speech_data, labels, pids = batch
                cough_data  = cough_data.to(torch.float32).to(device)
                speech_data = speech_data.to(torch.float32).to(device)
                labels      = labels.to(device)
                logits      = model(cough_data, speech_data)
                batch_size  = cough_data.size(0)
            elif len(batch) == 3:
                input_data, labels, pids = batch
                input_data = input_data.to(torch.float32).to(device)
                labels     = labels.to(device)
                logits     = model(input_data)
                batch_size = input_data.size(0)
            else:
                input_data, labels = batch
                pids       = None
                input_data = input_data.to(torch.float32).to(device)
                labels     = labels.to(device)
                logits     = model(input_data)
                batch_size = input_data.size(0)

            # ---------- per-cough loss (reference only) ----------
            cough_loss = criterion(logits, labels.long())
            cumulative_cough_loss += cough_loss.item() * batch_size
            total_coughs          += batch_size

            # ---------- group logits by patient ----------
            logits_cpu = logits.detach().cpu()
            if pids is not None:
                for i, pid in enumerate(pids):
                    patient_logits.setdefault(pid, []).append(logits_cpu[i])
                    patient_labels.setdefault(pid, labels[i].item())
            else:
                for i in range(batch_size):
                    key = f"sample_{sample_counter}"
                    sample_counter += 1
                    patient_logits.setdefault(key, []).append(logits_cpu[i])
                    patient_labels.setdefault(key, labels[i].item())

    # ---------- aggregate per patient in LOGIT space ----------
    agg_logits, agg_labels = [], []
    for pid, logits_list in patient_logits.items():
        mean_logit = torch.stack(logits_list).mean(0)   # [C]  (CPU)
        agg_logits.append(mean_logit)
        agg_labels.append(patient_labels[pid])

    # move to device so we can reuse the CUDA-resident class weights
    agg_logits = torch.stack(agg_logits).to(device)                 # [N_patients, C]
    agg_labels = torch.tensor(agg_labels).long().to(device)         # [N_patients]

    # ---------- per-patient loss ----------
    patient_loss = criterion(agg_logits, agg_labels).item()

    # ---------- per-patient accuracy / AUC (CPU for sklearn) ----------
    agg_probs_cpu  = torch.softmax(agg_logits, dim=1)[:, 1].detach().cpu()
    agg_labels_cpu = agg_labels.detach().cpu()

    predictions = (agg_probs_cpu > 0.5).float()
    acc         = (predictions == agg_labels_cpu).float().mean().item()

    if len(torch.unique(agg_labels_cpu)) > 1:
        fpr, tpr, _ = metrics.roc_curve(agg_labels_cpu, agg_probs_cpu)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = float("nan")

    cough_loss = cumulative_cough_loss / max(total_coughs, 1)

    if verbose:
        print(
            f"\n[{set_name.upper()}]  patients={len(agg_labels)}  "
            f"coughs={total_coughs}  "
            f"patient_loss={patient_loss:.4f}  cough_loss={cough_loss:.4f}  "
            f"acc={acc:.4f}  auc={auc:.4f}"
        )

    return patient_loss, acc, auc, cough_loss