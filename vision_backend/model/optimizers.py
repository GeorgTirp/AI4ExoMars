# optimizers.py
"""
Optimizer helpers for model training.

This module provides:

1. **create_optimizer**  
   Smart optimizer selection with priority:
   - Muon (if installed and explicitly enabled),
   - NAdam (PyTorch's NAdamW-like implementation),
   - AdamW as a safe fallback.

2. **create_cosine_scheduler_with_warmup**  
   A cosine–annealing learning rate schedule with linear warmup,
   mathematically equivalent to the Hugging Face transformers scheduler.

Both utilities are framework-agnostic and work with any PyTorch model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# Parameter-name fragments that must never receive weight decay even though
# they may be multi-dimensional (relative-position bias tables, GRN affine,
# the SimMIM mask fill scalar). 1-D params (norms/biases) are excluded by shape.
_NO_DECAY_NAME_FRAGMENTS = (
    "relative_position_bias_table",
    "mask_value",
    ".grn.",
)

# Try importing Muon (optional dependency)
try:
    from muon import Muon  # KellerJordan/Muon optimizer
    _HAS_MUON = True
except Exception:
    Muon = None  # type: ignore
    _HAS_MUON = False


# ---------------------------------------------------------------------------
# Optimizer Factory
# ---------------------------------------------------------------------------
def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    use_muon: bool = True,
) -> torch.optim.Optimizer:
    r"""
    Create an optimizer for a given model with prioritized fallback logic.

    The optimizers are tried in the following priority:

    1. **Muon** (if installed and ``use_muon=True``)  
       Muon is a second-order optimizer approximating natural gradient steps.

    2. **NAdam**  
       PyTorch's NAdam implementation (NadamW-style), supporting weight decay.

    3. **AdamW**  
       Stable, widely used, standard fallback.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose trainable parameters will be optimized.
    lr : float, optional
        Learning rate (default: ``3e-4``).
    weight_decay : float, optional
        Weight decay coefficient (default: ``1e-2``).
    use_muon : bool, optional
        Whether the user prefers to use Muon if available.

    Returns
    -------
    torch.optim.Optimizer
        Constructed optimizer instance.

    Notes
    -----
    - Only parameters with ``requires_grad=True`` are passed to the optimizer.
    - If Muon is requested but not installed, AdamW is used and a warning printed.
    """
    params_list = [p for p in model.parameters() if p.requires_grad]

    # --------------------
    # 1) Try Muon
    # --------------------
    if use_muon and _HAS_MUON:
        print("[optimizers] Using Muon optimizer.")
        return Muon(params_list, lr=lr, weight_decay=weight_decay)  # type: ignore

    # --------------------
    # 2) Try NAdam (NadamW-style)
    # --------------------
    if hasattr(torch.optim, "NAdam"):
        print("[optimizers] Using NAdam (NadamW-style) optimizer.")
        return torch.optim.NAdam(params_list, lr=lr, weight_decay=weight_decay)

    # --------------------
    # 3) Fallback: AdamW
    # --------------------
    if use_muon and not _HAS_MUON:
        print(
            "[optimizers] Muon requested but not installed.\n"
            "Install via: pip install git+https://github.com/KellerJordan/Muon"
        )

    print("[optimizers] Using AdamW optimizer.")
    return torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Cosine Annealing LR Schedule with Warmup
# ---------------------------------------------------------------------------
def create_cosine_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""
    Create a cosine-annealing LR scheduler with linear warmup.

    This scheduler combines a linear warmup phase with a cosine decay phase.

    **Learning rate schedule**

    Given current step :math:`t`, warmup :math:`W`, and total steps :math:`T`,
    the schedule is:

    **Warmup (linear)**

    .. math::

        \text{lr}(t) = \frac{t}{W}, \quad 0 \le t < W

    **Cosine decay**

    .. math::

        \text{progress} = \frac{t - W}{T - W}

        \text{lr}(t) =
        \tfrac{1}{2}\left(1 + \cos\big( 2\pi \cdot C \cdot \text{progress} \big)\right)

    where:

    - :math:`C` = ``num_cycles`` controls the number of cosine waves
      (``0.5`` = standard: decay → 0 once)

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    num_warmup_steps : int
        Number of linear warmup steps, typically 5–10% of total training steps.
    num_training_steps : int
        Total number of steps (``epochs * steps_per_epoch``).
    num_cycles : float, optional
        Number of cosine cycles.
        Default ``0.5`` = half-cycle (decay to 0 exactly once).

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler that updates the LR dynamically during training.

    Notes
    -----
    - This implementation is mathematically similar to
      ``transformers.get_cosine_schedule_with_warmup``.
    - The value returned by the lambda is multiplied with the optimizer's base LR.
    """

    def lr_lambda(current_step: int) -> float:
        # ---- Linear warmup ----
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)

        # ---- Cosine decay ----
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Stage-1 SimMIM: weight-decay param groups + optimizer + floored scheduler
# ---------------------------------------------------------------------------
def split_decay_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Two param groups: weight decay on >=2-D weights, none on the rest.

    Excludes norms/biases (1-D), relative-position-bias tables, GRN affine
    params, and the mask fill scalar from weight decay (§5).
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(frag in name for frag in _NO_DECAY_NAME_FRAGMENTS):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_simmim_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    optimizer: str = "adamw",
) -> torch.optim.Optimizer:
    """AdamW (default) with WD param groups, or a Muon-hybrid behind a flag.

    The Muon-hybrid path is best-effort: it requires the optional ``muon``
    package and falls back to AdamW (with a warning) when it is unavailable.
    """
    param_groups = split_decay_param_groups(model, weight_decay)

    if optimizer == "muon_hybrid":
        if _HAS_MUON:
            try:
                print("[optimizers] Using Muon-hybrid (Muon on 2-D weights, AdamW elsewhere).")
                param_groups[0]["use_muon"] = True
                param_groups[1]["use_muon"] = False
                return Muon(param_groups, lr=lr, betas=betas)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - depends on muon version
                print(f"[optimizers] Muon-hybrid unavailable ({exc}); falling back to AdamW.")
        else:
            print("[optimizers] Muon requested but not installed; using AdamW.")

    print("[optimizers] Using AdamW with weight-decay param groups.")
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def create_warmup_cosine_with_floor(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay from 1.0 down to ``min_lr_ratio``.

    ``min_lr_ratio = min_lr / base_lr`` so the LR floors at ``min_lr`` instead
    of decaying to zero.
    """
    min_lr_ratio = max(0.0, min(min_lr_ratio, 1.0))

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
