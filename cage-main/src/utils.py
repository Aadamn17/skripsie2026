import torch
import random
import numpy as np


def pad_to_224(img):
    """Centre-crop if larger than 224, centre-pad with zeros if smaller."""
    if img.ndim == 2:
        img = img.unsqueeze(0)
    H, W = img.shape[-2], img.shape[-1]

    if H > 224:
        top = (H - 224) // 2
        img = img[:, top:top + 224, :]
        H = 224
    if W > 224:
        left = (W - 224) // 2
        img = img[:, :, left:left + 224]
        W = 224

    # symmetric padding so signal stays centred
    pad_h_total = max(0, 224 - H)
    pad_w_total = max(0, 224 - W)
    pad_h_top   = pad_h_total // 2
    pad_h_bot   = pad_h_total - pad_h_top
    pad_w_left  = pad_w_total // 2
    pad_w_right = pad_w_total - pad_w_left

    if pad_h_total > 0 or pad_w_total > 0:
        img = torch.nn.functional.pad(
            img,
            (pad_w_left, pad_w_right, pad_h_top, pad_h_bot),
            "constant", 0,
        )
    return img[0]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def standard_collate(batch):
    images, labels, pids = zip(*batch)
    return (torch.stack(images),
            torch.tensor(labels, dtype=torch.long),
            list(pids))


def late_fusion_collate(batch):
    cough, speech, labels, pids = zip(*batch)
    return (torch.stack(cough),
            torch.stack(speech),
            torch.tensor(labels, dtype=torch.long),
            list(pids))