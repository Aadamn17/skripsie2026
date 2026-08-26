import torch
import torch.nn.functional as F

def exclusive_loss(z1, z2, labels, margin=1.0):
    batch_size = z1.size(0)
    dist = torch.cdist(z1, z2, p=2)                     # (batch, batch)
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # bool
    pos_mask = label_matrix.float()
    neg_mask = (~label_matrix).float()                  # correct inversion
    pos_loss = (pos_mask * dist).sum() / max(pos_mask.sum(), 1)
    neg_loss = (neg_mask * torch.clamp(margin - dist, min=0)).sum() / max(neg_mask.sum(), 1)
    return pos_loss + neg_loss


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    features = torch.cat([z1, z2], dim=0)
    sim = torch.mm(features, features.t()) / temperature
    mask = torch.eye(2*batch_size, device=features.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    loss = -torch.log(torch.exp(positives) / torch.exp(sim).sum(dim=1)).mean()
    return loss


def barlow_twins_loss(z1, z2, lambda_=0.005):
    batch_size = z1.size(0)
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-8)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-8)
    C = torch.mm(z1_norm.T, z2_norm) / batch_size
    invariance = ((1 - torch.diag(C))**2).sum()
    off_diag = C - torch.diag(torch.diag(C))
    redundancy = (off_diag**2).sum()
    loss = invariance + lambda_ * redundancy
    return loss / z1.size(1)


def pair_loss(z1, z2, labels, margin=1.0):
    dist = torch.norm(z1 - z2, dim=1)
    pos_mask = labels.float()
    neg_mask = 1 - pos_mask
    loss = (pos_mask * (dist**2) + neg_mask * torch.clamp(margin - dist, min=0)**2).mean()
    return loss