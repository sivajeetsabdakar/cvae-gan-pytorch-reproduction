from __future__ import annotations

import torch
from torch.nn import functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())


def mean_feature_matching(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.mse_loss(fake.mean(dim=0), real.mean(dim=0))


def class_feature_matching(
    real: torch.Tensor,
    fake: torch.Tensor,
    real_labels: torch.Tensor,
    fake_labels: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for label in torch.unique(torch.cat((real_labels, fake_labels))):
        real_mask = real_labels == label
        fake_mask = fake_labels == label
        if real_mask.any() and fake_mask.any():
            losses.append(F.mse_loss(fake[fake_mask].mean(0), real[real_mask].mean(0)))
    return 0.5 * torch.stack(losses).mean() if losses else fake.sum() * 0.0


def pairwise_reconstruction(
    real_image: torch.Tensor,
    reconstructed: torch.Tensor,
    real_d_feature: torch.Tensor,
    recon_d_feature: torch.Tensor,
    real_c_feature: torch.Tensor,
    recon_c_feature: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        F.mse_loss(reconstructed, real_image)
        + F.mse_loss(recon_d_feature, real_d_feature)
        + F.mse_loss(recon_c_feature, real_c_feature)
    )

