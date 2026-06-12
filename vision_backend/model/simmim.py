"""SimMIM masked-image-modeling head for the v2 hybrid encoder (§3).

SimMIM (not MAE): masking is done in pixel space (masked pixels are replaced
by a single learnable scalar) so the conv stem and Swin windows stay intact.
The decoder is intentionally minimal -- one 1x1 conv + PixelShuffle -- so that
representation quality lives in the encoder, not the decoder (guardrail §10).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_encoder import STAGE_DIMS


class Masker(nn.Module):
    """Random 32x32-unit pixel-space masking with a learnable fill scalar.

    Returns ``(x_masked, mask)`` where ``mask`` is a ``(B, H/unit, W/unit)``
    boolean tensor (True = masked). A precomputed mask may be supplied to make
    masking deterministic (used by tests and fixed-crop monitoring panels).
    """

    def __init__(self, mask_unit: int = 32, mask_ratio: float = 0.6):
        super().__init__()
        self.mask_unit = mask_unit
        self.mask_ratio = mask_ratio
        # single learnable scalar, init 0.0
        self.mask_value = nn.Parameter(torch.zeros(()))

    def sample_mask(self, batch: int, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        num = gh * gw
        k = int(round(num * self.mask_ratio))
        noise = torch.rand(batch, num, device=device)
        ids = noise.argsort(dim=1)
        mask_flat = torch.zeros(batch, num, dtype=torch.bool, device=device)
        if k > 0:
            mask_flat.scatter_(1, ids[:, :k], True)
        return mask_flat.view(batch, gh, gw)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if h % self.mask_unit != 0 or w % self.mask_unit != 0:
            raise ValueError(
                f"Input {h}x{w} must be divisible by mask unit {self.mask_unit}."
            )
        gh, gw = h // self.mask_unit, w // self.mask_unit

        if mask is None:
            mask = self.sample_mask(b, gh, gw, x.device)

        mask_px = mask.to(x.dtype).view(b, 1, gh, gw)
        mask_px = F.interpolate(mask_px, scale_factor=self.mask_unit, mode="nearest")
        x_masked = x * (1.0 - mask_px) + self.mask_value * mask_px
        return x_masked, mask


class ReconDecoder(nn.Module):
    """Minimal decoder: 1x1 conv (dim -> out*unit^2) + PixelShuffle(unit)."""

    def __init__(self, in_dim: int, out_channels: int = 1, mask_unit: int = 32):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_dim, out_channels * mask_unit * mask_unit, kernel_size=1)
        self.shuffle = nn.PixelShuffle(mask_unit)

    def forward(self, s4: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(s4))


class SimMIMModel(nn.Module):
    """Wraps an encoder with a SimMIM masker + reconstruction decoder + loss."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        in_channels: int = 1,
        encoder_dim: int = STAGE_DIMS[-1],
        mask_unit: int = 32,
        mask_ratio: float = 0.6,
        loss_type: str = "l1",
    ):
        super().__init__()
        if loss_type not in ("l1", "mse", "smooth_l1"):
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.encoder = encoder
        self.masker = Masker(mask_unit=mask_unit, mask_ratio=mask_ratio)
        self.decoder = ReconDecoder(encoder_dim, out_channels=in_channels, mask_unit=mask_unit)
        self.mask_unit = mask_unit
        self.loss_type = loss_type

    def _per_pixel_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return (recon - target).abs()
        if self.loss_type == "mse":
            return (recon - target).pow(2)
        return F.smooth_l1_loss(recon, target, reduction="none")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        target = x
        x_masked, mask = self.masker(x, mask=mask)
        feats = self.encoder(x_masked)
        recon = self.decoder(feats["s4"])

        mask_px = mask.to(recon.dtype).unsqueeze(1)
        mask_px = F.interpolate(mask_px, scale_factor=self.mask_unit, mode="nearest")

        per_px = self._per_pixel_loss(recon, target)
        denom = mask_px.sum().clamp(min=1.0) * recon.shape[1]
        loss = (per_px * mask_px).sum() / denom

        return {
            "loss": loss,
            "reconstruction": recon,
            "masked_input": x_masked,
            "mask": mask,
        }
