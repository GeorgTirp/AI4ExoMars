#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        description="Stage 1: self-supervised teacher pretraining on paired HiRISE local/context crops."
    )
    parser.add_argument("--index-path", default="data/hirise_context_pairs/patch_index.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--local-input-size", type=int, default=256)
    parser.add_argument("--context-input-size", type=int, default=256)
    parser.add_argument("--local-base-channels", type=int, default=48)
    parser.add_argument("--context-base-channels", type=int, default=24)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--mask-patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
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
    parser.add_argument(
        "--loss-type",
        choices=("l1", "mse", "smooth_l1"),
        default="l1",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--disable-stage32", action="store_true")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--checkpoint-path", default="checkpoints/stage1_teacher_ssl.pt")
    parser.add_argument("--history-path", default="outputs/stage1_teacher_ssl_history.csv")
    parser.add_argument("--examples-path", default="outputs/stage1_teacher_ssl_examples.pt")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_arguments(parser)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    return {
        "stage": "stage1_teacher_ssl",
        "seed": args.seed,
        "data": {
            "index_path": args.index_path,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "local_input_size": args.local_input_size,
            "context_input_size": args.context_input_size,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        },
        "model": {
            "in_channels": 1,
            "local_base_channels": args.local_base_channels,
            "context_base_channels": args.context_base_channels,
            "context_dim": args.context_dim,
            "decoder_channels": args.decoder_channels,
            "mask_patch_size": args.mask_patch_size,
            "mask_ratio": args.mask_ratio,
            "window_size": args.window_size,
            "drop_path": args.drop_path,
            "swin_depths": tuple(args.swin_depths),
            "swin_num_heads": tuple(args.swin_num_heads),
            "use_stage32": not args.disable_stage32,
            "loss_type": args.loss_type,
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
            "examples_path": args.examples_path,
            "num_examples": args.num_examples,
        },
    }


def train_stage(config: dict, wandb_run=None) -> dict:
    import torch
    from context_reconstruction import (
        collect_context_examples,
        create_context_patch_dataloaders,
        run_context_epoch,
    )
    from model.optimizers import (
        create_cosine_scheduler_with_warmup,
        create_optimizer,
    )
    from training.builders import build_context_pretrainer
    from training.utils import (
        count_parameters,
        resolve_path,
        save_checkpoint,
        save_history,
        select_device,
        set_seed,
    )

    set_seed(torch, int(config["seed"]))
    device = select_device(torch)
    use_amp = device.type == "cuda"

    data_config = config["data"]
    model_config = config["model"]
    optimization = config["optimization"]
    output = config["output"]

    checkpoint_path = resolve_path(output["checkpoint_path"])
    history_path = resolve_path(output["history_path"])
    examples_path = resolve_path(output["examples_path"])

    loaders = create_context_patch_dataloaders(
        index_path=resolve_path(data_config["index_path"]),
        batch_size=int(data_config["batch_size"]),
        local_input_size=int(data_config["local_input_size"]),
        context_input_size=int(data_config["context_input_size"]),
        val_fraction=float(data_config["val_fraction"]),
        test_fraction=float(data_config["test_fraction"]),
        num_workers=int(data_config["num_workers"]),
        seed=int(config["seed"]),
    )

    model = build_context_pretrainer(model_config).to(device)
    optimizer = create_optimizer(
        model,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        use_muon=bool(optimization["use_muon"]),
    )
    total_steps = int(optimization["epochs"]) * max(len(loaders.train), 1)
    warmup_steps = int(float(optimization["warmup_fraction"]) * total_steps)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Using device: {device}")
    print(f"Teacher params: {count_parameters(model):,}")
    print(f"Teacher trainable params: {count_parameters(model, trainable_only=True):,}")
    print(f"Train patches: {len(loaders.train_dataset)}")
    print(f"Val patches:   {len(loaders.val_dataset)}")

    history_rows: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, int(optimization["epochs"]) + 1):
        train_loss = run_context_epoch(
            model=model,
            dataloader=loaders.train,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            progress_desc=f"Stage1 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [train]",
            progress_position=0,
            leave_progress=True,
        )
        val_loss = run_context_epoch(
            model=model,
            dataloader=loaders.val,
            device=device,
            optimizer=None,
            use_amp=False,
            progress_desc=f"Stage1 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [val]",
            progress_position=0,
            leave_progress=True,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": current_lr,
        }
        history_rows.append(row)
        print(
            f"[stage1 epoch {epoch:02d}/{int(optimization['epochs']):02d}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.3e}"
        )
        log_metrics(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "optimizer/lr": current_lr,
            },
            step=epoch,
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            save_checkpoint(
                torch,
                {
                    "stage": config["stage"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "metrics": {"val_loss": val_loss},
                    "config": config,
                },
                checkpoint_path,
            )

    save_history(history_rows, history_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    examples = collect_context_examples(
        model=model,
        dataloader=loaders.val,
        device=device,
        num_examples=int(output["num_examples"]),
    )
    examples["best_val_loss"] = torch.tensor(best_val_loss)
    examples["best_epoch"] = torch.tensor(best_epoch)
    examples["checkpoint_path"] = str(checkpoint_path)
    save_checkpoint(torch, examples, examples_path)

    final_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": float(best_epoch),
        "total_params": float(count_parameters(model)),
        "trainable_params": float(count_parameters(model, trainable_only=True)),
    }
    log_metrics(wandb_run, final_metrics)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved history to {history_path}")
    print(f"Saved examples to {examples_path}")
    return final_metrics


def main() -> int:
    args = parse_args()
    base_config = build_config(args)

    if maybe_run_sweep(
        args,
        stage_name="stage1_teacher_ssl",
        base_config=base_config,
        train_fn=train_stage,
    ):
        return 0

    run = init_wandb_run(args, base_config, stage_name="stage1_teacher_ssl")
    try:
        merged_config = merge_wandb_config(base_config, run)
        train_stage(merged_config, run)
    finally:
        finish_wandb_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
