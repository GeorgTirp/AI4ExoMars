"""Typed, validated configuration for Stage-1 SimMIM training.

A single YAML file (configs/stage1_simmim.yaml) is the source of truth. It is
parsed into nested dataclasses with validation and a few derived quantities
(LR scaling, gradient-accumulation steps). Dotted-key overrides (e.g. from a
wandb sweep or ``--set optim.base_lr=2e-4``) can be applied before
construction via :func:`apply_overrides`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    index_path: str = "data/hirise_simmim_crops/crops_index.csv"
    splits_path: str | None = None
    batch_size: int = 8
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 2
    augment: bool = True
    crop_size: int = 1024


@dataclass
class ModelConfig:
    in_channels: int = 1
    global_base_grid: int = 32
    window_size: int = 8
    drop_path: float = 0.1
    use_checkpoint: bool = True
    mask_unit: int = 32
    mask_ratio: float = 0.6
    loss_type: str = "l1"


@dataclass
class OptimConfig:
    optimizer: str = "adamw"  # or "muon_hybrid"
    base_lr: float = 1.5e-4  # per effective batch of 256
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_fraction: float = 0.05
    effective_batch: int = 256
    grad_clip: float = 1.0
    total_steps: int = 200_000
    ema_decay: float = 0.9999


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints/stage1_simmim"
    run_name: str = "stage1_simmim"
    checkpoint_every: int = 5_000
    val_every: int = 1_000
    panel_every: int = 2_000
    log_every: int = 50
    num_panels: int = 8


@dataclass
class Stage1Config:
    seed: int = 42
    precision: str = "bf16"  # bf16 | fp32
    deterministic: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.data.crop_size % 256 != 0:
            raise ValueError("data.crop_size must be divisible by 256.")
        if self.data.crop_size // 32 != self.model.global_base_grid:
            raise ValueError(
                "model.global_base_grid must equal data.crop_size // 32 "
                f"({self.data.crop_size // 32}); got {self.model.global_base_grid}."
            )
        if not 0.0 < self.model.mask_ratio < 1.0:
            raise ValueError("model.mask_ratio must be in (0, 1).")
        if self.optim.effective_batch < self.data.batch_size:
            raise ValueError("optim.effective_batch must be >= data.batch_size.")
        if self.optim.optimizer not in ("adamw", "muon_hybrid"):
            raise ValueError(f"Unknown optimizer: {self.optim.optimizer}")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"Unknown precision: {self.precision}")

    # ---- derived quantities -------------------------------------------
    @property
    def grad_accum_steps(self) -> int:
        return max(1, round(self.optim.effective_batch / self.data.batch_size))

    @property
    def effective_batch(self) -> int:
        return self.data.batch_size * self.grad_accum_steps

    @property
    def scaled_lr(self) -> float:
        return self.optim.base_lr * self.effective_batch / 256.0

    @property
    def warmup_steps(self) -> int:
        return int(self.optim.warmup_fraction * self.optim.total_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(cls, data: dict[str, Any]):
    """Build a (possibly nested) dataclass from a dict, ignoring unknown keys."""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    for name, value in data.items():
        if name not in type_hints:
            raise ValueError(f"Unknown config key '{name}' for {cls.__name__}.")
        field_type = type_hints[name]
        if isinstance(value, dict) and is_dataclass(_resolve_dataclass(field_type)):
            kwargs[name] = _from_dict(_resolve_dataclass(field_type), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve_dataclass(field_type):
    mapping = {
        "DataConfig": DataConfig,
        "ModelConfig": ModelConfig,
        "OptimConfig": OptimConfig,
        "OutputConfig": OutputConfig,
    }
    if is_dataclass(field_type):
        return field_type
    return mapping.get(field_type, field_type)


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides (``optim.base_lr`` -> nested dict) in place."""
    for key, value in overrides.items():
        parts = key.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data


def config_from_dict(data: dict[str, Any]) -> Stage1Config:
    return _from_dict(Stage1Config, data)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Stage1Config:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = apply_overrides(data, overrides)
    return config_from_dict(data)
