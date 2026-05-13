#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train context-aware masked reconstruction on paired HiRISE patches."
    )
    parser.add_argument(
        "--index-path",
        default="data/hirise_patches/patch_index.csv",
        help="Path to the paired local/context patch index CSV.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--local-input-size", type=int, default=None)
    parser.add_argument("--context-input-size", type=int, default=256)
    parser.add_argument("--local-base-channels", type=int, default=48)
    parser.add_argument("--context-base-channels", type=int, default=24)
    parser.add_argument("--context-dim", type=int, default=256)
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
    )
    parser.add_argument(
        "--swin-num-heads",
        nargs=3,
        type=int,
        default=(4, 8, 16),
        metavar=("H8", "H16", "H32"),
    )
    parser.add_argument(
        "--loss-type",
        choices=("l1", "mse", "smooth_l1"),
        default="l1",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--disable-stage32",
        action="store_true",
        help="Use only the 1/8 and 1/16 Swin stages.",
    )
    parser.add_argument(
        "--use-muon",
        action="store_true",
        help="Prefer Muon if installed. Otherwise fall back automatically.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/best_context_masked_reconstruction.pt",
    )
    parser.add_argument(
        "--history-path",
        default="outputs/context_masked_reconstruction_history.csv",
    )
    parser.add_argument(
        "--examples-path",
        default="outputs/context_masked_reconstruction_examples.pt",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def select_device(torch_module):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def save_history(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        import torch

        from context_reconstruction import (
            collect_context_examples,
            create_context_patch_dataloaders,
            run_context_epoch,
        )
        from model.model import (
            ContextAwareConvNeXtSwinEncoder,
            ContextAwareMaskedReconstructionPretrainer,
        )
        from model.optimizers import (
            create_cosine_scheduler_with_warmup,
            create_optimizer,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(f"Unable to import the context reconstruction stack because `{missing}` is not installed.")
        return 1

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(torch)
    use_amp = device.type == "cuda"

    index_path = resolve_path(args.index_path)
    checkpoint_path = resolve_path(args.checkpoint_path)
    history_path = resolve_path(args.history_path)
    examples_path = resolve_path(args.examples_path)

    loaders = create_context_patch_dataloaders(
        index_path=index_path,
        batch_size=args.batch_size,
        local_input_size=args.local_input_size,
        context_input_size=args.context_input_size,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    use_stage32 = not args.disable_stage32
    encoder = ContextAwareConvNeXtSwinEncoder(
        in_channels=1,
        local_base_channels=args.local_base_channels,
        context_base_channels=args.context_base_channels,
        context_dim=args.context_dim,
        use_stage32=use_stage32,
        swin_depths=tuple(args.swin_depths),
        swin_num_heads=tuple(args.swin_num_heads),
        window_size=args.window_size,
    )
    bottleneck_channels = args.local_base_channels * (16 if use_stage32 else 8)
    skip8_channels = args.local_base_channels * 4

    model = ContextAwareMaskedReconstructionPretrainer(
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
    total_steps = args.epochs * max(len(loaders.train), 1)
    warmup_steps = int(args.warmup_fraction * total_steps)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history_rows: list[dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_context_epoch(
            model=model,
            dataloader=loaders.train,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
        )
        val_loss = run_context_epoch(
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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "metrics": {"val_loss": val_loss},
                    "config": vars(args),
                },
                checkpoint_path,
            )

    save_history(history_rows, history_path)
    print(f"Saved training history to {history_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    examples = collect_context_examples(
        model=model,
        dataloader=loaders.val,
        device=device,
        num_examples=args.num_examples,
    )
    examples["best_val_loss"] = torch.tensor(best_val_loss)
    examples["checkpoint_path"] = str(checkpoint_path)

    examples_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, examples_path)
    print(f"Saved example tensors to {examples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
