import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvggish import vggish as vggish_func

class VGGishEncoder(nn.Module):
    def __init__(self, trainable=False, device='cuda'):
        super().__init__()
        self.model = vggish_func()
        # Move entire model to the desired device
        self.model.to(device)
        # Move PCA parameters to the same device (fixes device mismatch)
        if hasattr(self.model, 'pproc'):
            self.model.pproc._pca_matrix = self.model.pproc._pca_matrix.to(device)
            self.model.pproc._pca_means = self.model.pproc._pca_means.to(device)
        # Freeze or unfreeze
        for param in self.model.parameters():
            param.requires_grad = False
        if trainable:
            for param in self.model.parameters():
                param.requires_grad = True
        self.feature_dim = 128
        self.device = device

    def forward(self, x):
        # Ensure input is on the same device as the model
        if x.device != self.device:
            x = x.to(self.device)
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x1, x2=None):
        if x2 is not None:
            z1 = self.projection_head(self.encoder(x1))
            z2 = self.projection_head(self.encoder(x2))
            return z1, z2
        else:
            return self.encoder(x1)

    def get_features(self, x):
        return self.encoder(x)