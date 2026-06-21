"""Exponential moving average of model weights for Stage-1 SimMIM.

The EMA shadow weights are what get evaluated and exported as the contract
artifact (``encoder_only.pt``). Buffers (e.g. none here, but kept for safety)
are copied directly; parameters are tracked with decay 0.9999.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # a frozen deep copy living on the same device as the source model
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        ema_params = dict(self.ema.named_parameters())
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ema_params[name].mul_(d).add_(param.detach(), alpha=1.0 - d)
        # keep buffers (running stats, rel-pos index, etc.) in sync
        ema_buffers = dict(self.ema.named_buffers())
        for name, buf in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buf)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.ema.parameters()

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state)

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema.to(device)
        return self
