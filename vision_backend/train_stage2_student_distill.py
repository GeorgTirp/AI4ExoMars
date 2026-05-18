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
        description="Stage 2: distill a smaller student encoder from the large SSL-pretrained teacher."
    )
    parser.add_argument("--index-path", default="data/hirise_context_pairs/patch_index.csv")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--local-input-size", type=int, default=256)
    parser.add_argument("--context-input-size", type=int, default=256)
    parser.add_argument("--student-local-base-channels", type=int, default=32)
    parser.add_argument("--student-context-base-channels", type=int, default=16)
    parser.add_argument("--student-context-dim", type=int, default=192)
    parser.add_argument("--student-decoder-channels", type=int, default=192)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument(
        "--student-swin-depths",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        metavar=("S8", "S16", "S32"),
    )
    parser.add_argument(
        "--student-swin-num-heads",
        nargs=3,
        type=int,
        default=(4, 8, 16),
        metavar=("H8", "H16", "H32"),
    )
    parser.add_argument("--disable-student-stage32", action="store_true")
    parser.add_argument(
        "--feature-indices",
        nargs="+",
        type=int,
        default=(1, 3, -1),
        help="Feature indices to distill from teacher to student.",
    )
    parser.add_argument(
        "--feature-weights",
        nargs="+",
        type=float,
        default=(1.0, 1.0, 1.0),
        help="Loss weights aligned with --feature-indices.",
    )
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Disable per-feature normalization before distillation loss.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--checkpoint-path", default="checkpoints/stage2_student_distill.pt")
    parser.add_argument("--history-path", default="outputs/stage2_student_distill_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_arguments(parser)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    return {
        "stage": "stage2_student_distill",
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
        "teacher": {
            "checkpoint_path": args.teacher_checkpoint,
        },
        "student_model": {
            "in_channels": 1,
            "local_base_channels": args.student_local_base_channels,
            "context_base_channels": args.student_context_base_channels,
            "context_dim": args.student_context_dim,
            "decoder_channels": args.student_decoder_channels,
            "mask_patch_size": 16,
            "mask_ratio": 0.6,
            "window_size": args.window_size,
            "drop_path": args.drop_path,
            "swin_depths": tuple(args.student_swin_depths),
            "swin_num_heads": tuple(args.student_swin_num_heads),
            "use_stage32": not args.disable_student_stage32,
            "loss_type": "l1",
        },
        "distillation": {
            "feature_indices": [int(index) for index in args.feature_indices],
            "feature_weights": [float(weight) for weight in args.feature_weights],
            "normalize_features": not args.no_normalize_features,
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


def train_stage(config: dict, wandb_run=None) -> dict:
    import torch
    from context_reconstruction import create_context_patch_dataloaders, run_context_epoch
    from model.optimizers import (
        create_cosine_scheduler_with_warmup,
        create_optimizer,
    )
    from training.builders import build_context_encoder, build_context_pretrainer
    from training.distillation import FeatureDistillationWrapper
    from training.utils import (
        count_parameters,
        infer_context_feature_channels,
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
    teacher_config = config["teacher"]
    student_model_config = config["student_model"]
    distillation_config = config["distillation"]
    optimization = config["optimization"]
    output = config["output"]

    checkpoint_path = resolve_path(output["checkpoint_path"])
    history_path = resolve_path(output["history_path"])
    teacher_checkpoint_path = resolve_path(teacher_config["checkpoint_path"])

    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu")
    if "config" not in teacher_checkpoint or "model" not in teacher_checkpoint["config"]:
        raise KeyError(
            "Teacher checkpoint must contain the stage1 config with a `model` section."
        )
    teacher_model_config = dict(teacher_checkpoint["config"]["model"])
    teacher_model = build_context_pretrainer(teacher_model_config)
    teacher_model.load_state_dict(teacher_checkpoint["model_state"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    student_encoder = build_context_encoder(student_model_config).to(device)

    teacher_feature_channels = infer_context_feature_channels(
        torch,
        teacher_model.encoder,
        local_input_size=int(data_config["local_input_size"]),
        context_input_size=int(data_config["context_input_size"]),
        in_channels=int(student_model_config["in_channels"]),
    )
    student_feature_channels = infer_context_feature_channels(
        torch,
        student_encoder,
        local_input_size=int(data_config["local_input_size"]),
        context_input_size=int(data_config["context_input_size"]),
        in_channels=int(student_model_config["in_channels"]),
    )

    wrapper = FeatureDistillationWrapper(
        teacher_encoder=teacher_model.encoder,
        student_encoder=student_encoder,
        teacher_feature_channels=[
            teacher_feature_channels[index]
            for index in distillation_config["feature_indices"]
        ],
        student_feature_channels=[
            student_feature_channels[index]
            for index in distillation_config["feature_indices"]
        ],
        feature_indices=distillation_config["feature_indices"],
        feature_weights=distillation_config["feature_weights"],
        normalize_features=bool(distillation_config["normalize_features"]),
    ).to(device)

    optimizer = create_optimizer(
        wrapper,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        use_muon=bool(optimization["use_muon"]),
    )

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

    total_steps = int(optimization["epochs"]) * max(len(loaders.train), 1)
    warmup_steps = int(float(optimization["warmup_fraction"]) * total_steps)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Using device: {device}")
    print(f"Teacher encoder params: {count_parameters(teacher_model.encoder):,}")
    print(f"Student encoder params: {count_parameters(student_encoder):,}")
    print(f"Distillation wrapper trainable params: {count_parameters(wrapper, trainable_only=True):,}")
    print(f"Train patches: {len(loaders.train_dataset)}")
    print(f"Val patches:   {len(loaders.val_dataset)}")

    history_rows: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, int(optimization["epochs"]) + 1):
        train_loss = run_context_epoch(
            model=wrapper,
            dataloader=loaders.train,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            progress_desc=f"Stage2 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [train]",
            progress_position=0,
            leave_progress=True,
        )
        val_loss = run_context_epoch(
            model=wrapper,
            dataloader=loaders.val,
            device=device,
            optimizer=None,
            use_amp=False,
            progress_desc=f"Stage2 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [val]",
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
            f"[stage2 epoch {epoch:02d}/{int(optimization['epochs']):02d}] "
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
                    "model_state": wrapper.state_dict(),
                    "student_state": student_encoder.state_dict(),
                    "teacher_checkpoint_path": str(teacher_checkpoint_path),
                    "metrics": {"val_loss": val_loss},
                    "config": config,
                },
                checkpoint_path,
            )

    save_history(history_rows, history_path)
    final_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": float(best_epoch),
        "student_total_params": float(count_parameters(student_encoder)),
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
        stage_name="stage2_student_distill",
        base_config=base_config,
        train_fn=train_stage,
    ):
        return 0

    run = init_wandb_run(args, base_config, stage_name="stage2_student_distill")
    try:
        merged_config = merge_wandb_config(base_config, run)
        train_stage(merged_config, run)
    finally:
        finish_wandb_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
