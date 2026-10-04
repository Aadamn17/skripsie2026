import torch
import random
import numpy as np

# ImageNet channel statistics, used to normalise the 3-channel input
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


def min_max_rescale(image, eps=1e-8):
    """Rescale a tensor to [0, 1] using its own min and max."""
    vmin = image.min()
    vmax = image.max()
    return (image - vmin) / (vmax - vmin + eps)


def imagenet_normalize(image):
    """Normalise a 3-channel tensor with ImageNet statistics."""
    mean = IMAGENET_MEAN.to(image.device)
    std  = IMAGENET_STD.to(image.device)
    return (image - mean) / std


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