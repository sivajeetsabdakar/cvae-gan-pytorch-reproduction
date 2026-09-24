from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from cvaegan.models import CVAEGAN


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a class-conditioned image grid")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("samples.png"))
    parser.add_argument("--per-class", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    latent_dim = int(checkpoint["args"].get("latent_dim", 256))
    model = CVAEGAN(len(classes), latent_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    labels = torch.arange(len(classes), device=device).repeat_interleave(args.per_class)
    with torch.no_grad():
        images = model.sample(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image((images + 1) / 2, args.output, nrow=args.per_class)
    print(f"Saved {len(images)} images for {len(classes)} classes to {args.output}")


if __name__ == "__main__":
    main()

