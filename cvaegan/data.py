from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_loader(
    root: str | Path,
    batch_size: int,
    workers: int = 4,
    image_size: int = 128,
) -> tuple[DataLoader, list[str]]:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 16),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    dataset = datasets.ImageFolder(str(root), transform=transform)
    if len(dataset.classes) < 2:
        raise ValueError("The dataset must contain at least two class directories")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset.classes

