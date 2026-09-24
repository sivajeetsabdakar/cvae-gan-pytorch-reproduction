from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizers: dict, epoch: int, classes: list[str], args: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "classes": classes,
            "args": args,
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps({"epoch": epoch, "classes": classes, "args": args}, indent=2))

