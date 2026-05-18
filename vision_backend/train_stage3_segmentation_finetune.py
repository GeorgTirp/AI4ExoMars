#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from training.wandb_utils import (
    add_wandb_arguments,
    finish_wandb_run,
    init_wandb_run,
    log_metrics,
    maybe_run_sweep,
    merge_wandb_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: supervised segmentation fine-tuning of the distilled encoder."
    )
    parser.add_argument(
        "--loader-factory",
        default="martian_terrain_segmentation.dataloader:create_ai4mars_dataloaders",
        help="Import path to a loader factory returning train/val/test loaders.",
    )
    parser.add_argument(
        "--loader-config-path",
        default=None,
        help="Optional JSON config file forwarded to the loader factory.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--strict-checkpoint-load", action="store_true")
    parser.add_argument("--local-base-channels", type=int, default=32)
    parser.add_argument("--context-base-channels", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--decoder-channels", type=int, default=192)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--drop-path", type=float, default=0.0)
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
    parser.add_argument("--disable-stage32", action="store_true")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--checkpoint-path", default="checkpoints/stage3_segmentation.pt")
    parser.add_argument("--history-path", default="outputs/stage3_segmentation_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_arguments(parser)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    return {
        "stage": "stage3_segmentation_finetune",
        "seed": args.seed,
        "data": {
            "loader_factory": args.loader_factory,
            "loader_config_path": args.loader_config_path,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_classes": args.num_classes,
            "ignore_index": args.ignore_index,
        },
        "model": {
            "in_channels": 1,
            "local_base_channels": args.local_base_channels,
            "context_base_channels": args.context_base_channels,
            "context_dim": args.context_dim,
            "decoder_channels": args.decoder_channels,
            "window_size": args.window_size,
            "drop_path": args.drop_path,
            "swin_depths": tuple(args.swin_depths),
            "swin_num_heads": tuple(args.swin_num_heads),
            "use_stage32": not args.disable_stage32,
        },
        "initialization": {
            "encoder_checkpoint": args.encoder_checkpoint,
            "strict_checkpoint_load": args.strict_checkpoint_load,
            "freeze_encoder_epochs": args.freeze_encoder_epochs,
        },
        "optimization": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_fraction": args.warmup_fraction,
            "use_muon": args.use_muon,
        },
        "output": {
            "checkpoint_path": args.checkpoint_path,
            "history_path": args.history_path,
        },
    }


def _load_loader_kwargs(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    path = resolve_path(config_path)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise TypeError("Loader config JSON must deserialize to a dictionary.")
    return loaded


def train_stage(config: dict, wandb_run=None) -> dict:
    import torch
    from model.optimizers import (
        create_cosine_scheduler_with_warmup,
        create_optimizer,
    )
    from training.builders import (
        build_context_segmentation_model,
        load_encoder_from_pretrainer_checkpoint,
    )
    from training.utils import (
        count_parameters,
        freeze_module,
        load_loader_bundle,
        resolve_path,
        run_segmentation_epoch,
        save_checkpoint,
        save_history,
        select_device,
        set_seed,
        unfreeze_module,
    )

    set_seed(torch, int(config["seed"]))
    device = select_device(torch)
    use_amp = device.type == "cuda"

    data_config = config["data"]
    model_config = dict(config["model"])
    initialization = config["initialization"]
    optimization = config["optimization"]
    output = config["output"]

    loader_kwargs = _load_loader_kwargs(data_config["loader_config_path"])
    loader_kwargs.update(
        {
            "batch_size": int(data_config["batch_size"]),
            "num_workers": int(data_config["num_workers"]),
            "seed": int(config["seed"]),
        }
    )
    loaders = load_loader_bundle(data_config["loader_factory"], loader_kwargs)
    if "train" not in loaders or "val" not in loaders:
        raise KeyError("Loader bundle must contain at least `train` and `val` loaders.")

    num_classes = data_config.get("num_classes")
    if num_classes is None:
        num_classes = loaders.get("num_classes")
    if num_classes is None:
        raise ValueError(
            "num_classes was not provided and could not be inferred from the loader bundle."
        )
    model_config["num_classes"] = int(num_classes)

    model = build_context_segmentation_model(model_config).to(device)
    load_encoder_from_pretrainer_checkpoint(
        torch,
        model.encoder,
        resolve_path(initialization["encoder_checkpoint"]),
        strict=bool(initialization["strict_checkpoint_load"]),
    )

    optimizer = create_optimizer(
        model,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        use_muon=bool(optimization["use_muon"]),
    )
    total_steps = int(optimization["epochs"]) * max(len(loaders["train"]), 1)
    warmup_steps = int(float(optimization["warmup_fraction"]) * total_steps)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    checkpoint_path = resolve_path(output["checkpoint_path"])
    history_path = resolve_path(output["history_path"])

    print(f"Using device: {device}")
    print(f"Segmentation model params: {count_parameters(model):,}")
    print(f"Encoder params: {count_parameters(model.encoder):,}")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches:   {len(loaders['val'])}")

    history_rows: list[dict[str, float]] = []
    best_val_miou = float("-inf")
    best_epoch = 0
    encoder_unfrozen = False

    freeze_encoder_epochs = int(initialization["freeze_encoder_epochs"])
    if freeze_encoder_epochs > 0:
        freeze_module(model.encoder)

    for epoch in range(1, int(optimization["epochs"]) + 1):
        if freeze_encoder_epochs > 0 and epoch > freeze_encoder_epochs and not encoder_unfrozen:
            unfreeze_module(model.encoder)
            encoder_unfrozen = True
            print(f"[stage3] Unfroze encoder at epoch {epoch}.")

        train_metrics = run_segmentation_epoch(
            torch,
            model,
            loaders["train"],
            device,
            num_classes=int(num_classes),
            ignore_index=int(data_config["ignore_index"]),
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            progress_desc=f"Stage3 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [train]",
            leave_progress=True,
        )
        val_metrics = run_segmentation_epoch(
            torch,
            model,
            loaders["val"],
            device,
            num_classes=int(num_classes),
            ignore_index=int(data_config["ignore_index"]),
            optimizer=None,
            use_amp=False,
            progress_desc=f"Stage3 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [val]",
            leave_progress=True,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_pixel_acc": float(train_metrics["pixel_acc"]),
            "train_miou": float(train_metrics["miou"]),
            "val_loss": float(val_metrics["loss"]),
            "val_pixel_acc": float(val_metrics["pixel_acc"]),
            "val_miou": float(val_metrics["miou"]),
            "lr": current_lr,
        }
        history_rows.append(row)
        log_metrics(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/pixel_acc": train_metrics["pixel_acc"],
                "train/miou": train_metrics["miou"],
                "val/loss": val_metrics["loss"],
                "val/pixel_acc": val_metrics["pixel_acc"],
                "val/miou": val_metrics["miou"],
                "optimizer/lr": current_lr,
            },
            step=epoch,
        )

        print(
            f"[stage3 epoch {epoch:02d}/{int(optimization['epochs']):02d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_miou={val_metrics['miou']:.6f} "
            f"lr={current_lr:.3e}"
        )

        if float(val_metrics["miou"]) > best_val_miou:
            best_val_miou = float(val_metrics["miou"])
            best_epoch = epoch
            save_checkpoint(
                torch,
                {
                    "stage": config["stage"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metrics": dict(val_metrics),
                    "config": config,
                },
                checkpoint_path,
            )

    save_history(history_rows, history_path)
    final_metrics = {
        "best_val_miou": best_val_miou,
        "best_epoch": float(best_epoch),
        "total_params": float(count_parameters(model)),
    }
    log_metrics(wandb_run, final_metrics)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved history to {history_path}")
    return final_metrics


def main() -> int:
    args = parse_args()
    base_config = build_config(args)

    if maybe_run_sweep(
        args,
        stage_name="stage3_segmentation_finetune",
        base_config=base_config,
        train_fn=train_stage,
    ):
        return 0

    run = init_wandb_run(args, base_config, stage_name="stage3_segmentation_finetune")
    try:
        merged_config = merge_wandb_config(base_config, run)
        train_stage(merged_config, run)
    finally:
        finish_wandb_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
