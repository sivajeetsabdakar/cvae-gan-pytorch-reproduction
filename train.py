from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from cvaegan.data import build_loader
from cvaegan.losses import (
    class_feature_matching,
    kl_divergence,
    mean_feature_matching,
    pairwise_reconstruction,
)
from cvaegan.models import CVAEGAN
from cvaegan.utils import save_checkpoint, seed_everything, set_requires_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVAE-GAN on an ImageFolder dataset")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with one directory per class")
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-kl", type=float, default=3.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-d-match", type=float, default=1e-3)
    parser.add_argument("--lambda-c-match", type=float, default=1e-3)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N updates; 0 runs all epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, classes = build_loader(args.data, args.batch_size, args.workers)
    model = CVAEGAN(len(classes), args.latent_dim).to(device)

    optimizers = {
        "encoder": Adam(model.encoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999)),
        "generator": Adam(model.generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)),
        "discriminator": Adam(model.discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)),
        "classifier": Adam(model.classifier.parameters(), lr=args.lr, betas=(args.beta1, 0.999)),
    }
    bce = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(args.output / "tensorboard")
    sample_dir = args.output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    amp_enabled = args.amp and device.type == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp_enabled else nullcontext

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for real, labels in progress:
            real, labels = real.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = real.size(0)
            ones, zeros = torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)

            # Train classifier on real class labels.
            optimizers["classifier"].zero_grad(set_to_none=True)
            with autocast():
                class_logits, _ = model.classifier(real)
                classifier_loss = F.cross_entropy(class_logits, labels)
            classifier_loss.backward()
            optimizers["classifier"].step()

            # Train discriminator on real, reconstructed, and prior samples.
            with torch.no_grad():
                latent, _, _ = model.encode(real, labels)
                reconstructed_detached = model.generator(latent, labels)
                prior_labels = torch.randint(0, len(classes), (batch_size,), device=device)
                prior_detached = model.sample(prior_labels)
            optimizers["discriminator"].zero_grad(set_to_none=True)
            with autocast():
                real_logits, _ = model.discriminator(real)
                recon_logits, _ = model.discriminator(reconstructed_detached)
                prior_logits, _ = model.discriminator(prior_detached)
                discriminator_loss = bce(real_logits, ones) + bce(recon_logits, zeros) + bce(prior_logits, zeros)
            discriminator_loss.backward()
            optimizers["discriminator"].step()

            # Train encoder using KL regularization and pairwise reconstruction.
            set_requires_grad(model.discriminator, False)
            set_requires_grad(model.classifier, False)
            optimizers["encoder"].zero_grad(set_to_none=True)
            with autocast():
                latent, mu, logvar = model.encode(real, labels)
                reconstructed = model.generator(latent, labels)
                with torch.no_grad():
                    _, real_d_feature = model.discriminator(real)
                    _, real_c_feature = model.classifier(real)
                _, recon_d_feature = model.discriminator(reconstructed)
                _, recon_c_feature = model.classifier(reconstructed)
                reconstruction_loss = pairwise_reconstruction(
                    real, reconstructed, real_d_feature, recon_d_feature, real_c_feature, recon_c_feature
                )
                encoder_loss = args.lambda_kl * kl_divergence(mu, logvar) + args.lambda_recon * reconstruction_loss
            encoder_loss.backward()
            optimizers["encoder"].step()

            # Train generator with pairwise and class-conditional mean matching.
            optimizers["generator"].zero_grad(set_to_none=True)
            with autocast():
                with torch.no_grad():
                    latent, _, _ = model.encode(real, labels)
                    _, real_d_feature = model.discriminator(real)
                    _, real_c_feature = model.classifier(real)
                reconstructed = model.generator(latent.detach(), labels)
                prior_labels = torch.randint(0, len(classes), (batch_size,), device=device)
                prior = model.sample(prior_labels)
                _, recon_d_feature = model.discriminator(reconstructed)
                _, recon_c_feature = model.classifier(reconstructed)
                _, prior_d_feature = model.discriminator(prior)
                _, prior_c_feature = model.classifier(prior)
                reconstruction_loss_g = pairwise_reconstruction(
                    real, reconstructed, real_d_feature, recon_d_feature, real_c_feature, recon_c_feature
                )
                d_match = mean_feature_matching(real_d_feature, prior_d_feature)
                c_match = class_feature_matching(real_c_feature, prior_c_feature, labels, prior_labels)
                generator_loss = (
                    args.lambda_recon * reconstruction_loss_g
                    + args.lambda_d_match * d_match
                    + args.lambda_c_match * c_match
                )
            generator_loss.backward()
            optimizers["generator"].step()
            set_requires_grad(model.discriminator, True)
            set_requires_grad(model.classifier, True)

            losses = {
                "classifier": classifier_loss.item(),
                "discriminator": discriminator_loss.item(),
                "encoder": encoder_loss.item(),
                "generator": generator_loss.item(),
                "kl": kl_divergence(mu.detach(), logvar.detach()).item(),
            }
            for name, value in losses.items():
                writer.add_scalar(f"loss/{name}", value, global_step)
            progress.set_postfix(g=f"{losses['generator']:.3f}", d=f"{losses['discriminator']:.3f}")

            if global_step % args.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    fixed_labels = torch.arange(min(len(classes), 16), device=device)
                    samples = model.sample(fixed_labels)
                    save_image((samples + 1) / 2, sample_dir / f"step-{global_step:07d}.png", nrow=4)
                model.train()
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                args.output / "checkpoints" / f"epoch-{epoch:04d}.pt",
                model,
                optimizers,
                epoch,
                classes,
                vars(args) | {"data": str(args.data), "output": str(args.output)},
            )
        if args.max_steps and global_step >= args.max_steps:
            break

    writer.close()


if __name__ == "__main__":
    main()
