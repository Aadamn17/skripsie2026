import os
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import torch.nn as nn
from torch.amp import autocast
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMP_KWARGS = dict(device_type="cuda", dtype=torch.bfloat16)


class EarlyStopper:
    def __init__(self, patience=7, min_delta=1e-3, mode="max",
                 window=3, min_epochs=10):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.window, self.min_epochs = window, min_epochs
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter, self.best_epoch = 0, 0
        self.best_state = None
        self.should_stop = False
        self.history = []

    def step(self, value, model, epoch):
        if value is None or (isinstance(value, float) and value != value):
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.should_stop = True
            return False

        self.history.append(value)
        if len(self.history) < self.window:
            return False
        smoothed = sum(self.history[-self.window:]) / self.window

        improved = (smoothed < self.best - self.min_delta if self.mode == "min"
                    else smoothed > self.best + self.min_delta)

        if improved:
            self.best, self.counter, self.best_epoch = smoothed, 0, epoch
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
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.3, use_pretrained=True):
        super().__init__()
        self.use_pretrained = use_pretrained
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.resnet = resnet18(weights=weights)

        if use_pretrained:
            for p in self.resnet.parameters():
                p.requires_grad = False
            for name, p in self.resnet.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    p.requires_grad = True

        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        # keep frozen BN layers in eval mode
        if self.use_pretrained and mode:
            for m in self.resnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        return self.resnet(x)


class LateFusion(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True, dropout_p=0.3):
        super().__init__()
        self.use_pretrained = use_pretrained
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.cough_backbone = resnet18(weights=weights)
        self.cough_backbone.fc = nn.Linear(512, num_classes)

        if use_pretrained:
            for p in self.cough_backbone.parameters():
                p.requires_grad = False
            for name, p in self.cough_backbone.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    p.requires_grad = True

        self.speech_lr   = Logistic_Regression(input_dim=128, num_classes=num_classes)
        self.fusion_head = nn.Linear(2 * num_classes, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.use_pretrained and mode:
            for m in self.cough_backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, cough, speech):
        cough_logits  = self.cough_backbone(cough)
        speech_logits = self.speech_lr(speech)
        fused = torch.cat([cough_logits, speech_logits], dim=1)
        return self.fusion_head(fused)

    def param_groups(self, base_lr, weight_decay):
        # speech branch and fusion head both use 10x base_lr
        return [
            {"params": self.cough_backbone.parameters(),
             "lr": base_lr,      "weight_decay": weight_decay},
            {"params": self.speech_lr.parameters(),
             "lr": base_lr * 10, "weight_decay": weight_decay},
            {"params": self.fusion_head.parameters(),
             "lr": base_lr * 10, "weight_decay": weight_decay},
        ]


def _save_predictions(dev_preds, test_preds, params, predictions_dir):
    """Persist per-patient predictions for downstream analysis."""
    os.makedirs(predictions_dir, exist_ok=True)
    rows = [{**p, "set": "dev"} for p in dev_preds]
    rows += [{**p, "set": "test"} for p in test_preds]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["fusion"]     = params["fusion"]
    df["arch"]       = params["arch"]
    df["lr"]         = params["learning_rate"]
    df["wd"]         = params["weight_decay"]
    df["aug"]        = params["augmentation"]
    df["pretrained"] = params["use_pretrained"]
    df["test_set"]   = params["test_set"]
    df["dev_set"]    = params["dev_set"]

    fname = (
        f"fusion-{params['fusion']}_arch-{params['arch']}_"
        f"pretrained-{str(params['use_pretrained']).lower()}_"
        f"aug-{params['augmentation']}_lr-{params['learning_rate']}_"
        f"wd-{params['weight_decay']}_"
        f"test-{params['test_set']}_dev-{params['dev_set']}.csv"
    )
    df.to_csv(os.path.join(predictions_dir, fname), index=False)


def train_validate(train_data, dev_data, test_data, model, params,
                   predictions_dir=None):
    if hasattr(model, "param_groups"):
        optimizer = torch.optim.AdamW(
            model.param_groups(params["learning_rate"], params["weight_decay"])
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["num_epochs"]
    )

    # class weights from the underlying datasets
    all_labels = []
    if hasattr(train_data.dataset, "datasets"):
        for ds in train_data.dataset.datasets:
            if hasattr(ds, "samples"):
                all_labels.extend([s[2] for s in ds.samples])
            elif hasattr(ds, "labels"):
                all_labels.extend(ds.labels["Status"].tolist())
            elif hasattr(ds, "df"):
                all_labels.extend(ds.df["Status"].tolist())
    class_counts = torch.bincount(
        torch.tensor(all_labels), minlength=2
    ).clamp(min=1)
    weights = (class_counts.sum() / (2 * class_counts.float())).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    stopper = EarlyStopper(
        patience=params.get("early_stop_patience", 7),
        min_delta=params.get("early_stop_min_delta", 5e-3),
        mode="max",
        window=params.get("early_stop_window", 5),
        min_epochs=params.get("early_stop_min_epochs", 10),
    )

    history, best_epoch, final_test_metrics = [], None, None
    print(f"\n{'='*120}\nFold: test={params.get('test_set')} "
          f"dev={params.get('dev_set')}\n{'-'*120}")

    for epoch in range(params["num_epochs"]):
        train_loss = train_epoch(train_data, model, optimizer, criterion)
        scheduler.step()

        if dev_data is not None:
            dev_loss_pat, dev_acc, dev_auc, dev_sens, dev_spec, dev_loss_cough = \
                evaluate_epoch(dev_data, model, criterion,
                               set_name="dev", verbose=False)
            is_best = stopper.step(dev_auc, model, epoch)
        else:
            dev_loss_pat = dev_loss_cough = float("inf")
            dev_acc = dev_auc = dev_sens = dev_spec = float("nan")
            is_best = False

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "dev_loss_patient": dev_loss_pat, "dev_loss_cough": dev_loss_cough,
            "dev_acc": dev_acc, "dev_auc": dev_auc,
            "dev_sens": dev_sens, "dev_spec": dev_spec,
            "test_loss_patient": float("nan"), "test_loss_cough": float("nan"),
            "test_acc": float("nan"), "test_auc": float("nan"),
            "test_sens": float("nan"), "test_spec": float("nan"),
            "is_best": int(is_best),
        })

        print(f"{epoch+1:>4} | tr_loss: {train_loss:>8.4f} "
              f"| dev_auc: {dev_auc:>8.4f} | dev_sens: {dev_sens:>8.4f} "
              f"| dev_spec: {dev_spec:>8.4f} |{' *best*' if is_best else ''}")

        if stopper.should_stop:
            break

    if dev_data is not None and stopper.best_state is not None:
        stopper.restore_best(model)
        best_epoch = stopper.best_epoch + 1
        print(f"\n[FINAL] restored best epoch = {best_epoch}")

        dev_preds = []
        evaluate_epoch(dev_data, model, criterion, set_name="dev_best",
                       verbose=True, predictions_out=dev_preds)

        test_preds = []
        if test_data is not None:
            tl_pat, ta, t_auc, t_sens, t_spec, tl_cough = evaluate_epoch(
                test_data, model, criterion, set_name="test_best",
                verbose=True, predictions_out=test_preds,
            )
            final_test_metrics = {
                "test_loss_patient": tl_pat, "test_loss_cough": tl_cough,
                "test_acc": ta, "test_auc": t_auc,
                "test_sens": t_sens, "test_spec": t_spec,
            }

        for h in history:
            if h["epoch"] == best_epoch and final_test_metrics:
                h.update(final_test_metrics)

        if predictions_dir is not None:
            _save_predictions(dev_preds, test_preds, params, predictions_dir)

    return history


def train_epoch(train_data, model, optimizer, criterion):
    model.train()
    cumulative_loss, total_samples = 0, 0

    for batch in train_data:
        optimizer.zero_grad(set_to_none=True)

        with autocast(**AMP_KWARGS):
            if len(batch) == 4:
                cough_data, speech_data, labels, _ = batch
                cough_data  = cough_data.to(torch.float32).to(device)
                speech_data = speech_data.to(torch.float32).to(device)
                output      = model(cough_data, speech_data)
                batch_size  = cough_data.size(0)
            else:
                input_data, labels = batch[0], batch[1]
                input_data = input_data.to(torch.float32).to(device)
                output     = model(input_data)
                batch_size = input_data.size(0)

            loss = criterion(output, labels.long().to(device))

        cumulative_loss += loss.item() * batch_size
        total_samples   += batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return cumulative_loss / total_samples


def evaluate_epoch(data, model, criterion, set_name="eval", verbose=True,
                   predictions_out=None):
    model.eval()
    cumulative_cough_loss, total_coughs = 0.0, 0
    patient_logits, patient_labels, patient_n_coughs = {}, {}, {}
    sample_counter = 0

    with torch.inference_mode():
        for batch in data:
            with autocast(**AMP_KWARGS):
                if len(batch) == 4:
                    cough_data, speech_data, labels, pids = batch
                    cough_data  = cough_data.to(torch.float32).to(device)
                    speech_data = speech_data.to(torch.float32).to(device)
                    logits      = model(cough_data, speech_data)
                    batch_size  = cough_data.size(0)
                elif len(batch) == 3:
                    input_data, labels, pids = batch
                    input_data = input_data.to(torch.float32).to(device)
                    logits     = model(input_data)
                    batch_size = input_data.size(0)
                else:
                    input_data, labels = batch
                    pids       = None
                    input_data = input_data.to(torch.float32).to(device)
                    logits     = model(input_data)
                    batch_size = input_data.size(0)

            labels = labels.long().to(device)
            cough_loss = criterion(logits.float(), labels)
            cumulative_cough_loss += cough_loss.item() * batch_size
            total_coughs += batch_size

            logits_cpu = logits.float().detach().cpu()
            if pids is not None:
                for i, pid in enumerate(pids):
                    patient_logits.setdefault(pid, []).append(logits_cpu[i])
                    patient_labels.setdefault(pid, labels[i].item())
                    patient_n_coughs[pid] = patient_n_coughs.get(pid, 0) + 1
            else:
                for i in range(batch_size):
                    key = f"sample_{sample_counter}"
                    patient_logits.setdefault(key, []).append(logits_cpu[i])
                    patient_labels.setdefault(key, labels[i].item())
                    patient_n_coughs[key] = patient_n_coughs.get(key, 0) + 1
                    sample_counter += 1

    patient_ids = list(patient_logits.keys())
    agg_logits = torch.stack(
        [torch.stack(patient_logits[pid]).mean(0) for pid in patient_ids]
    ).to(device)
    agg_labels = torch.tensor(
        [patient_labels[pid] for pid in patient_ids]
    ).long().to(device)

    patient_loss   = criterion(agg_logits, agg_labels).item()
    agg_probs_cpu  = torch.softmax(agg_logits, dim=1)[:, 1].detach().cpu()
    agg_labels_cpu = agg_labels.detach().cpu()
    agg_logits_cpu = agg_logits.detach().cpu()

    predictions = (agg_probs_cpu > 0.5).float()
    acc = (predictions == agg_labels_cpu).float().mean().item()

    if len(torch.unique(agg_labels_cpu)) > 1:
        fpr, tpr, _ = metrics.roc_curve(agg_labels_cpu, agg_probs_cpu)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(
            agg_labels_cpu, predictions
        ).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    else:
        auc = sens = spec = float("nan")

    cough_loss = cumulative_cough_loss / max(total_coughs, 1)

    if verbose:
        print(f"\n[{set_name.upper()}] pat_loss={patient_loss:.4f} "
              f"acc={acc:.4f} auc={auc:.4f} sens={sens:.4f} spec={spec:.4f}")

    if predictions_out is not None:
        for i, pid in enumerate(patient_ids):
            predictions_out.append({
                "patient_id": str(pid),
                "true_label": int(agg_labels_cpu[i].item()),
                "pred_label": int(predictions[i].item()),
                "prob_pos":   float(agg_probs_cpu[i].item()),
                "logit_neg":  float(agg_logits_cpu[i, 0].item()),
                "logit_pos":  float(agg_logits_cpu[i, 1].item()),
                "n_coughs":   int(patient_n_coughs[pid]),
            })

    return patient_loss, acc, auc, sens, spec, cough_loss