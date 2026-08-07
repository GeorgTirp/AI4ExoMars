#!/usr/bin/env python3
"""Stage-1 SimMIM self-supervised pretraining for the v2 hybrid encoder.

Step-based loop with bf16 autocast, gradient checkpointing (S1-S3), gradient
accumulation to a large effective batch, global-norm clipping, weight EMA, a
floored cosine LR schedule, exact resume, and SSL monitoring (recon loss +
fixed-crop reconstruction panels). Probe/kNN evaluation is intentionally out of
scope for this round.

Usage:
    python -m vision_backend.train_stage1_simmim --config vision_backend/configs/stage1_simmim.yaml
    ... --set optim.base_lr=2e-4 --set data.batch_size=16
    ... --resume checkpoints/stage1_simmim/last.pt
"""

from __future__ import annotations

import argparse
import contextlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from vision_backend.model.hybrid_encoder import HybridEncoder
    from vision_backend.model.simmim import SimMIMModel
    from vision_backend.model.optimizers import (
        build_simmim_optimizer,
        create_warmup_cosine_with_floor,
    )
    from vision_backend.simmim_dataset import ResumableLoader, create_simmim_dataloaders
    from vision_backend.training.config import Stage1Config, config_from_dict, load_config
    from vision_backend.training.ema import ModelEMA
    from vision_backend.training.utils import (
        count_parameters,
        resolve_path,
        save_checkpoint,
        select_device,
        set_seed,
    )
    from vision_backend.training.wandb_utils import (
        add_wandb_arguments,
        finish_wandb_run,
        init_wandb_run,
        log_metrics,
        maybe_run_sweep,
        merge_wandb_config,
        validate_wandb_args,
    )
except ModuleNotFoundError:  # running as a script from inside vision_backend/
    from model.hybrid_encoder import HybridEncoder
    from model.simmim import SimMIMModel
    from model.optimizers import build_simmim_optimizer, create_warmup_cosine_with_floor
    from simmim_dataset import ResumableLoader, create_simmim_dataloaders
    from training.config import Stage1Config, config_from_dict, load_config
    from training.ema import ModelEMA
    from training.utils import (
        count_parameters,
        resolve_path,
        save_checkpoint,
        select_device,
        set_seed,
    )
    from training.wandb_utils import (
        add_wandb_arguments,
        finish_wandb_run,
        init_wandb_run,
        log_metrics,
        maybe_run_sweep,
        merge_wandb_config,
        validate_wandb_args,
    )


# ---------------------------------------------------------------------------
# Model assembly + reporting
# ---------------------------------------------------------------------------
def build_model(cfg: Stage1Config) -> SimMIMModel:
    encoder = HybridEncoder(
        in_channels=cfg.model.in_channels,
        global_base_grid=cfg.model.global_base_grid,
        window_size=cfg.model.window_size,
        drop_path=cfg.model.drop_path,
        use_checkpoint=cfg.model.use_checkpoint,
    )
    return SimMIMModel(
        encoder,
        in_channels=cfg.model.in_channels,
        mask_unit=cfg.model.mask_unit,
        mask_ratio=cfg.model.mask_ratio,
        loss_type=cfg.model.loss_type,
    )


def print_param_table(model: SimMIMModel) -> int:
    encoder = model.encoder
    print("[model] per-component parameters:")
    for name, n in encoder.param_table():
        print(f"  {name:16s} {n:>12,}")
    enc_total = count_parameters(encoder)
    print(f"  {'decoder':16s} {count_parameters(model.decoder):>12,}")
    print(f"  {'total model':16s} {count_parameters(model):>12,}")
    if not (26_000_000 <= enc_total <= 31_000_000):
        raise AssertionError(
            f"Encoder param count {enc_total:,} outside the contract budget [26M, 31M]."
        )
    return enc_total


# ---------------------------------------------------------------------------
# Precision / determinism helpers
# ---------------------------------------------------------------------------
def autocast_context(device: torch.device, precision: str):
    if precision == "bf16" and device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def configure_determinism(deterministic: bool) -> None:
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Evaluation + monitoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(eval_model: nn.Module, val_loader, device: torch.device, precision: str) -> float:
    eval_model.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        x = batch["image"].to(device).float()
        with autocast_context(device, precision):
            out = eval_model(x)
        total += out["loss"].item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def _to_display(t: torch.Tensor) -> np.ndarray:
    # (1, H, W) in [-1, 1] -> (H, W) in [0, 1]
    return ((t.detach().float().cpu().squeeze(0).clamp(-1, 1) + 1.0) / 2.0).numpy()


@torch.no_grad()
def build_panels(eval_model: nn.Module, crops: torch.Tensor, masks: torch.Tensor, device: torch.device, precision: str):
    eval_model.eval()
    x = crops.to(device).float()
    with autocast_context(device, precision):
        out = eval_model(x, mask=masks.to(device))
    panels = []
    try:
        import wandb
    except ModuleNotFoundError as exc:
        # reachable only when a run is active, i.e. wandb was importable at init
        raise ModuleNotFoundError("wandb vanished mid-run; cannot log panels.") from exc
    for i in range(x.size(0)):
        triplet = np.concatenate(
            [
                _to_display(x[i]),
                _to_display(out["masked_input"][i]),
                _to_display(out["reconstruction"][i]),
            ],
            axis=1,
        )
        panels.append(wandb.Image(triplet, caption=f"crop{i}: input | masked | recon"))
    return panels


def make_fixed_masks(model: SimMIMModel, crops: torch.Tensor, seed: int = 1234) -> torch.Tensor:
    b, _, h, w = crops.shape
    gh, gw = h // model.mask_unit, w // model.mask_unit
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    masks = model.masker.sample_mask(b, gh, gw, torch.device("cpu"))
    torch.set_rng_state(rng_state)
    return masks


# ---------------------------------------------------------------------------
# Checkpoint state (full, exact-resume) + encoder-only artifact
# ---------------------------------------------------------------------------
def gather_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_full_checkpoint(path: Path, *, cfg, step, model, ema, optimizer, scheduler, loader) -> None:
    save_checkpoint(
        torch,
        {
            "stage": "stage1_simmim",
            "step": step,
            "config": cfg.to_dict(),
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loader_state": loader.state_dict(),
            "rng_state": gather_rng_state(),
        },
        path,
    )


def save_encoder_only(path: Path, ema: ModelEMA, cfg) -> None:
    # EMA weights are the contract artifact; strip to the encoder submodule.
    ema_state = ema.state_dict()
    prefix = "encoder."
    encoder_state = {
        k[len(prefix):]: v for k, v in ema_state.items() if k.startswith(prefix)
    }
    save_checkpoint(
        torch,
        {"stage": "stage1_simmim", "encoder_state": encoder_state, "model_config": cfg.model.__dict__},
        path,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    cfg: Stage1Config,
    args,
    *,
    max_steps: int | None = None,
    wandb_run=None,
) -> dict:
    set_seed(torch, cfg.seed)
    configure_determinism(cfg.deterministic)
    # deterministic test mode runs on CPU for bitwise-reproducible resume
    device = torch.device("cpu") if cfg.deterministic else select_device(torch)
    print(f"[stage1] device={device} precision={cfg.precision} "
          f"eff_batch={cfg.effective_batch} accum={cfg.grad_accum_steps} lr={cfg.scaled_lr:.2e}")

    index_path = resolve_path(cfg.data.index_path)
    splits_path = resolve_path(cfg.data.splits_path) if cfg.data.splits_path else None
    loaders = create_simmim_dataloaders(
        index_path,
        splits_path=splits_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        augment=cfg.data.augment,
        seed=cfg.seed,
    )
    train_loader = ResumableLoader(
        loaders.train_dataset,
        batch_size=cfg.data.batch_size,
        seed=cfg.seed,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=cfg.data.prefetch_factor,
    )
    print(f"[stage1] train crops={len(loaders.train_dataset)} "
          f"val crops={len(loaders.val_dataset) if loaders.val_dataset else 0}")

    model = build_model(cfg).to(device)
    print_param_table(model)
    ema = ModelEMA(model, decay=cfg.optim.ema_decay).to(device)

    optimizer = build_simmim_optimizer(
        model,
        lr=cfg.scaled_lr,
        weight_decay=cfg.optim.weight_decay,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        optimizer=cfg.optim.optimizer,
    )
    scheduler = create_warmup_cosine_with_floor(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.optim.total_steps,
        min_lr_ratio=cfg.optim.min_lr / max(cfg.scaled_lr, 1e-12),
    )

    global_step = 0
    if args.resume:
        ckpt = torch.load(resolve_path(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        train_loader.load_state_dict(ckpt["loader_state"])
        restore_rng_state(ckpt["rng_state"])
        global_step = int(ckpt["step"])
        print(f"[stage1] resumed from {args.resume} at step {global_step}")

    # In sweep mode the agent owns the run (already initialized with the sampled
    # params); standalone runs create their own.
    owns_run = wandb_run is None
    run = wandb_run
    if owns_run and not args.no_wandb:
        run = init_wandb_run(args, cfg.to_dict(), stage_name="stage1_simmim")

    checkpoint_dir = resolve_path(cfg.output.checkpoint_dir)
    if run is not None and not owns_run:
        # one dir per trial -- concurrent agents would otherwise clobber last.pt
        checkpoint_dir = checkpoint_dir / f"run_{run.id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage1] checkpoint_dir={checkpoint_dir}")

    if not owns_run and loaders.val is None:
        # sweeps optimize val/loss; with no val split the metric is never logged
        # and every trial would look equally (un)good to the bayes optimizer.
        raise RuntimeError(
            "Sweep trial has no validation split, so the sweep metric 'val/loss' "
            "would never be logged. Rebuild the crop index with a non-empty val "
            "set (prep_crops --val-fraction/--min-val-observations)."
        )

    # fixed val crops + masks for reproducible monitoring panels
    panel_crops = panel_masks = None
    if loaders.val_dataset is not None and len(loaders.val_dataset) > 0:
        n = min(cfg.output.num_panels, len(loaders.val_dataset))
        panel_crops = torch.stack([loaders.val_dataset[i]["image"] for i in range(n)])
        panel_masks = make_fixed_masks(model, panel_crops)

    data_iter = iter(train_loader)
    total_steps = cfg.optim.total_steps  # drives the LR schedule
    stop_step = total_steps if max_steps is None else min(max_steps, total_steps)
    history: list[dict] = []
    model.train()

    while global_step < stop_step:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = next(data_iter)
            x = batch["image"].to(device, non_blocking=True).float()
            with autocast_context(device, cfg.precision):
                out = model(x)
                loss = out["loss"] / cfg.grad_accum_steps
            loss.backward()
            accum_loss += out["loss"].item()

        if cfg.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)
        global_step += 1

        train_loss = accum_loss / cfg.grad_accum_steps
        lr = optimizer.param_groups[0]["lr"]
        history.append({"step": global_step, "train_loss": train_loss, "lr": lr})

        if global_step % cfg.output.log_every == 0:
            print(f"[stage1] step {global_step}/{total_steps} train_loss={train_loss:.6f} lr={lr:.3e}")
            log_metrics(run, {"train/loss": train_loss, "optimizer/lr": lr}, step=global_step)

        if loaders.val is not None and global_step % cfg.output.val_every == 0:
            val_loss = evaluate(ema.ema, loaders.val, device, cfg.precision)
            print(f"[stage1] step {global_step} val_loss={val_loss:.6f}")
            log_metrics(run, {"val/loss": val_loss}, step=global_step)
            model.train()

        if run is not None and panel_crops is not None and global_step % cfg.output.panel_every == 0:
            panels = build_panels(ema.ema, panel_crops, panel_masks, device, cfg.precision)
            if panels:
                log_metrics(run, {"val/reconstructions": panels}, step=global_step)
            model.train()

        if global_step % cfg.output.checkpoint_every == 0 or global_step == stop_step:
            save_full_checkpoint(
                checkpoint_dir / "last.pt",
                cfg=cfg, step=global_step, model=model, ema=ema,
                optimizer=optimizer, scheduler=scheduler, loader=train_loader,
            )
            save_encoder_only(checkpoint_dir / "encoder_only.pt", ema, cfg)
            print(f"[stage1] checkpoint @ step {global_step} -> {checkpoint_dir}")

    if owns_run:
        finish_wandb_run(run)  # in sweep mode the agent closes the run
    print(f"[stage1] done at step {global_step}.")
    return {"final_step": global_step, "history": history}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 SimMIM pretraining (v2 hybrid encoder).")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "stage1_simmim.yaml"),
        help="Path to the Stage-1 YAML config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Dotted-key config override, e.g. --set optim.base_lr=2e-4 (repeatable).",
    )
    parser.add_argument("--resume", default=None, help="Path to a full checkpoint to resume from.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb regardless of flags.")
    add_wandb_arguments(parser)
    return parser.parse_args()


def _coerce(value: str):
    import yaml
    return yaml.safe_load(value)


def main() -> int:
    args = parse_args()
    # fail before any expensive setup if the wandb/sweep flags are inconsistent
    validate_wandb_args(args)

    overrides = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = _coerce(value)
    cfg = load_config(args.config, overrides=overrides or None)

    def _train_from_dict(config_dict: dict, run=None) -> dict:
        # sweep params arrive as dotted keys already merged into config_dict;
        # config_from_dict rejects unknown keys loudly rather than ignoring them
        return train(config_from_dict(config_dict), args, wandb_run=run)

    if maybe_run_sweep(
        args,
        stage_name="stage1_simmim",
        base_config=cfg.to_dict(),
        train_fn=_train_from_dict,
    ):
        return 0

    train(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
