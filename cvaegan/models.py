from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int, *, normalize: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)
    ]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.features = nn.Sequential(
            _conv_block(3, 64, normalize=False),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, 512),
            _conv_block(512, 512),
        )
        self.projection = nn.Linear(512 * 4 * 4 + embedding_dim, 1024)
        self.mu = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)

    def forward(self, image: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(image).flatten(1)
        hidden = F.relu(self.projection(torch.cat((feature, self.embedding(labels)), dim=1)))
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class Generator(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.project = nn.Linear(latent_dim + embedding_dim, 512 * 4 * 4)
        self.network = nn.Sequential(
            _deconv_block(512, 512),
            _deconv_block(512, 256),
            _deconv_block(256, 128),
            _deconv_block(128, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = self.project(torch.cat((latent, self.embedding(labels)), dim=1))
        return self.network(hidden.view(latent.size(0), 512, 4, 4))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64, normalize=False),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, 512),
            _conv_block(512, 512),
        )
        self.head = nn.Linear(512 * 4 * 4, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(image).flatten(1)
        return self.head(feature).squeeze(1), feature


class Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64, normalize=False),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, 512),
            _conv_block(512, 512),
        )
        self.head = nn.Linear(512 * 4 * 4, num_classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(image).flatten(1)
        return self.head(feature), feature


class CVAEGAN(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.encoder = Encoder(num_classes, latent_dim)
        self.generator = Generator(num_classes, latent_dim)
        self.discriminator = Discriminator()
        self.classifier = Classifier(num_classes)

    def encode(self, image: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(image, labels)
        return self.encoder.reparameterize(mu, logvar), mu, logvar

    def sample(self, labels: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        if latent is None:
            latent = torch.randn(labels.size(0), self.latent_dim, device=labels.device)
        return self.generator(latent, labels)

