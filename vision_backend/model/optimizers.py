# optimizers.py
"""Optimizer helpers for model training."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch

_MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _orthogonalize_newton_schulz(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
) -> torch.Tensor:
    """Muon Newton-Schulz orthogonalization for a 2D update matrix."""
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D tensor.")

    original_dtype = matrix.dtype
    compute_dtype = torch.bfloat16 if matrix.is_cuda else torch.float32
    update = matrix.to(compute_dtype)
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.mT

    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = _MUON_NS_COEFFS
    for _ in range(steps):
        gram = update @ update.mT
        update = a * update + (b * gram + c * gram @ gram) @ update

    if transposed:
        update = update.mT
    return update.to(original_dtype)


def _muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    momentum: float,
    ns_steps: int,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - momentum)
    update = grad.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    update = _orthogonalize_newton_schulz(update, steps=ns_steps)
    update = update * max(1.0, update.shape[0] / max(update.shape[1], 1)) ** 0.5
    return update


class MuonAdamW(torch.optim.Optimizer):
    """Single-device Muon for 2D matrices plus AdamW for the remaining params.

    The Muon update follows the KellerJordan/Muon optimizer design
    (MIT License, Copyright (c) 2024 Keller Jordan):
    https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        param_groups: list[dict],
        *,
        muon_ns_steps: int = 5,
    ):
        if not param_groups:
            raise ValueError("MuonAdamW requires at least one parameter group.")

        defaults = {
            "lr": 3e-4,
            "weight_decay": 1e-2,
            "use_muon": False,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "momentum": 0.95,
            "muon_ns_steps": muon_ns_steps,
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if group.get("use_muon", False):
                momentum = group["momentum"]
                ns_steps = group["muon_ns_steps"]
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.ndim != 2:
                        raise ValueError("Muon parameter groups may only contain 2D tensors.")

                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)

                    update = _muon_update(
                        grad,
                        state["momentum_buffer"],
                        momentum=momentum,
                        ns_steps=ns_steps,
                    )
                    if weight_decay != 0:
                        param.mul_(1.0 - lr * weight_decay)
                    param.add_(update, alpha=-lr)
                continue

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0:
                    param.mul_(1.0 - lr * weight_decay)

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss


def _numel(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(param.numel() for param in params)


# ---------------------------------------------------------------------------
# Optimizer Factory
# ---------------------------------------------------------------------------
def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    use_muon: bool = True,
) -> torch.optim.Optimizer:
    """Create AdamW, or Muon-on-2D-plus-AdamW for all other parameters."""
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    if not named_params:
        raise ValueError("Model has no trainable parameters.")

    if not use_muon:
        params = [param for _, param in named_params]
        print(f"[optimizers] Using AdamW for {len(params)} tensors ({_numel(params):,} params).")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    muon_params = [
        param
        for _, param in named_params
        if param.ndim == 2
    ]
    adamw_params = [
        param
        for _, param in named_params
        if param.ndim != 2
    ]

    param_groups: list[dict] = []
    if muon_params:
        param_groups.append(
            {
                "params": sorted(muon_params, key=lambda param: param.numel(), reverse=True),
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": 0.95,
                "muon_ns_steps": 5,
                "use_muon": True,
            }
        )
    if adamw_params:
        param_groups.append(
            {
                "params": adamw_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "use_muon": False,
            }
        )

    print(
        "[optimizers] Using MuonAdamW: "
        f"Muon handles {len(muon_params)} 2D tensors ({_numel(muon_params):,} params); "
        f"AdamW handles {len(adamw_params)} other tensors ({_numel(adamw_params):,} params)."
    )
    return MuonAdamW(param_groups)


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
