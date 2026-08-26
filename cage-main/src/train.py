import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from losses import exclusive_loss, nt_xent_loss, barlow_twins_loss, pair_loss
from evaluate import extract_features

def compute_val_loss(model, val_loader, loss_name, margin, temperature, lambda_, device):
    """Compute average contrastive loss on validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for view1, view2, labels in val_loader:
            view1 = view1.to(device).float()
            view2 = view2.to(device).float()
            labels = labels.to(device)
            z1, z2 = model(view1, view2)
            if loss_name == 'exclusive':
                loss = exclusive_loss(z1, z2, labels, margin=margin)
            elif loss_name == 'nt_xent':
                loss = nt_xent_loss(z1, z2, temperature=temperature)
            elif loss_name == 'barlow':
                loss = barlow_twins_loss(z1, z2, lambda_=lambda_)
            elif loss_name == 'pair':
                loss = pair_loss(z1, z2, labels, margin=margin)
            else:
                raise ValueError(f"Unknown loss: {loss_name}")
            total_loss += loss.item()
    return total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')


def train_contrastive(train_loader, val_loader, model, loss_name, epochs, lr, weight_decay,
                      margin=1.0, temperature=0.5, lambda_=0.005, patience=10, device='cuda'):
    """
    Train Siamese network contrastively.
    Early stopping based on validation loss (not AUC).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for view1, view2, labels in train_loader:
            view1 = view1.to(device).float()
            view2 = view2.to(device).float()
            labels = labels.to(device)

            z1, z2 = model(view1, view2)

            if loss_name == 'exclusive':
                loss = exclusive_loss(z1, z2, labels, margin=margin)
            elif loss_name == 'nt_xent':
                loss = nt_xent_loss(z1, z2, temperature=temperature)
            elif loss_name == 'barlow':
                loss = barlow_twins_loss(z1, z2, lambda_=lambda_)
            elif loss_name == 'pair':
                loss = pair_loss(z1, z2, labels, margin=margin)
            else:
                raise ValueError(f"Unknown loss: {loss_name}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loss
        if val_loader is not None and len(val_loader) > 0:
            val_loss = compute_val_loss(model, val_loader, loss_name, margin, temperature, lambda_, device)
        else:
            val_loss = float('inf')

        scheduler.step(val_loss)  # mode='min'

        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping based on validation loss (lower is better)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss