#!/usr/bin/env python3
"""
Minimal masked reconstruction pretraining for the ConvNeXt+Swin hybrid encoder.

This script trains the encoder with the reconstruction wrapper defined in
`martian_terrain_segmentation.exo_models` and writes:

- a CSV training history
- a best-model checkpoint
- a small tensor bundle with a few original / masked / reconstructed examples
- a PNG grid of those reconstruction examples
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train minimal masked reconstruction pretraining on AI4Mars."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument(
        "--swin-depths",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        metavar=("S8", "S16", "S32"),
        help="Depth of the Swin stages at 1/8, 1/16, and 1/32.",
    )
    parser.add_argument(
        "--swin-num-heads",
        nargs=3,
        type=int,
        default=(4, 8, 16),
        metavar=("H8", "H16", "H32"),
        help="Attention heads at 1/8, 1/16, and 1/32.",
    )
    parser.add_argument(
        "--loss-type",
        choices=("l1", "mse", "smooth_l1"),
        default="l1",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging runs.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging runs.",
    )
    parser.add_argument(
        "--scan-spurious",
        action="store_true",
        help="Scan and cache valid AI4Mars indices on this run.",
    )
    parser.add_argument(
        "--use-muon",
        action="store_true",
        help="Prefer Muon if installed. Otherwise falls back automatically.",
    )
    parser.add_argument(
        "--disable-stage32",
        action="store_true",
        help="Use only the 1/8 and 1/16 Swin stages.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-disk-path",
        default="data/ai4mars_hf",
        help="Path of the on-disk Hugging Face dataset copy.",
    )
    parser.add_argument(
        "--valid-indices-cache-dir",
        default="ai4mars_valid_indices",
        help="Directory for cached valid sample indices.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/best_masked_reconstruction.pt",
        help="Best-model checkpoint output path.",
    )
    parser.add_argument(
        "--history-path",
        default="outputs/masked_reconstruction_history.csv",
        help="CSV training history output path.",
    )
    parser.add_argument(
        "--examples-path",
        default="outputs/masked_reconstruction_examples.pt",
        help="Saved reconstruction examples output path.",
    )
    parser.add_argument(
        "--examples-png-path",
        default="outputs/masked_reconstruction_examples.png",
        help="PNG grid for quick visual inspection of reconstruction examples.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="How many examples to save for notebook inspection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device(torch_module) -> "torch_module.device":
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def run_epoch(model, dataloader, device, optimizer=None, scheduler=None, use_amp=False):
    import torch

    training = optimizer is not None
    model.train(training)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for imgs, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True).float()
            batch_size = imgs.size(0)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(imgs)
                loss = outputs["loss"]

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_history(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)


def collect_examples(model, dataloader, device, num_examples: int) -> dict:
    import torch

    model.eval()
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True).float()
            outputs = model(imgs)
            count = min(num_examples, imgs.size(0))
            return {
                "original": imgs[:count].detach().cpu(),
                "masked_input": outputs["masked_input"][:count].detach().cpu(),
                "reconstruction": outputs["reconstruction"][:count].detach().cpu(),
                "mask": outputs["mask"][:count].detach().cpu(),
            }

    raise RuntimeError("Unable to collect reconstruction examples from an empty dataloader.")


def save_examples_png(examples: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    original = examples["original"]
    masked_input = examples["masked_input"]
    reconstruction = examples["reconstruction"].clamp(0.0, 1.0)
    mask = examples["mask"]
    num_examples = original.shape[0]

    fig, axes = plt.subplots(
        num_examples,
        4,
        figsize=(12, 3 * num_examples),
        squeeze=False,
    )

    for idx in range(num_examples):
        axes[idx, 0].imshow(original[idx, 0].numpy(), cmap="gray")
        axes[idx, 0].set_title(f"Original {idx + 1}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(masked_input[idx, 0].numpy(), cmap="gray")
        axes[idx, 1].set_title("Masked Input")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(reconstruction[idx, 0].numpy(), cmap="gray")
        axes[idx, 2].set_title("Reconstruction")
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(mask[idx, 0].numpy(), cmap="magma")
        axes[idx, 3].set_title("Mask")
        axes[idx, 3].axis("off")

    best_val_loss = examples.get("best_val_loss")
    if best_val_loss is not None:
        fig.suptitle(
            f"Masked Reconstruction Examples | best val loss = {best_val_loss:.6f}",
            y=1.01,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    sys.path.insert(0, str(SRC_DIR))

    try:
        import torch
        from martian_terrain_segmentation.dataloader import create_ai4mars_dataloaders
        from martian_terrain_segmentation.exo_models import (
            ConvNeXtSwinEncoder,
            MaskedReconstructionPretrainer,
        )
        from martian_terrain_segmentation.optimizers import (
            create_cosine_scheduler_with_warmup,
            create_optimizer,
        )
        from martian_terrain_segmentation.train_utils import load_checkpoint, save_checkpoint
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            "Unable to import the reconstruction training stack because "
            f"`{missing}` is not installed.",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(torch)
    use_amp = device.type == "cuda"

    checkpoint_path = _resolve_path(args.checkpoint_path)
    history_path = _resolve_path(args.history_path)
    examples_path = _resolve_path(args.examples_path)
    examples_png_path = _resolve_path(args.examples_png_path)
    local_disk_path = _resolve_path(args.local_disk_path)
    valid_indices_cache_dir = _resolve_path(args.valid_indices_cache_dir)
    cache_dir = _resolve_path(args.cache_dir) if args.cache_dir else None

    print(f"Using device: {device}")
    print("Loading AI4Mars dataloaders...")
    loaders = create_ai4mars_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_fraction=0.1,
        to_rgb=False,
        seed=args.seed,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_local_disk_copy=True,
        local_disk_path=str(local_disk_path),
        scan_spurious=args.scan_spurious,
        valid_indices_cache_dir=str(valid_indices_cache_dir),
    )

    use_stage32 = not args.disable_stage32
    encoder = ConvNeXtSwinEncoder(
        in_channels=1,
        base_channels=args.base_channels,
        use_stage32=use_stage32,
        swin_depths=tuple(args.swin_depths),
        swin_num_heads=tuple(args.swin_num_heads),
        window_size=args.window_size,
    )
    bottleneck_channels = args.base_channels * (16 if use_stage32 else 8)
    skip8_channels = args.base_channels * 4

    model = MaskedReconstructionPretrainer(
        encoder=encoder,
        in_channels=1,
        bottleneck_channels=bottleneck_channels,
        decoder_channels=args.decoder_channels,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        bottleneck_index=-1,
        skip8_index=3,
        skip8_channels=skip8_channels,
        use_skip8=True,
        loss_type=args.loss_type,
    ).to(device)

    optimizer = create_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        use_muon=args.use_muon,
    )
    total_steps = args.epochs * len(loaders.train)
    warmup_steps = int(args.warmup_fraction * total_steps)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=loaders.train,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
        )
        val_loss = run_epoch(
            model=model,
            dataloader=loaders.val,
            device=device,
            optimizer=None,
            use_amp=False,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        print(
            f"[epoch {epoch:02d}/{args.epochs:02d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"lr={current_lr:.3e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                path=str(checkpoint_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"val_loss": val_loss},
                extra={
                    "base_channels": args.base_channels,
                    "use_stage32": use_stage32,
                    "swin_depths": list(args.swin_depths),
                    "swin_num_heads": list(args.swin_num_heads),
                    "window_size": args.window_size,
                    "patch_size": args.patch_size,
                    "mask_ratio": args.mask_ratio,
                    "loss_type": args.loss_type,
                },
            )

    save_history(history_rows, history_path)
    print(f"Saved training history to {history_path}")

    load_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=None,
        scheduler=None,
        map_location=device,
    )
    examples = collect_examples(
        model=model,
        dataloader=loaders.val,
        device=device,
        num_examples=args.num_examples,
    )
    examples["best_val_loss"] = best_val_loss
    examples["checkpoint_path"] = str(checkpoint_path)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, examples_path)
    print(f"Saved reconstruction examples to {examples_path}")
    save_examples_png(examples, examples_png_path)
    print(f"Saved reconstruction example grid to {examples_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
