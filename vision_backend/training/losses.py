"""Segmentation losses for Stage-3, including long-tail corrections.

NOAH-H's own experiments found naive inverse-frequency class weighting a mixed
bag -- it lifted the boosted classes but hurt others. Logit adjustment (Menon et
al. 2021, "Long-tail learning via logit adjustment") is the cleaner fix: instead
of rescaling the loss per class, it shifts the logits by ``tau * log(prior_c)``
so the model learns the *balanced* posterior directly. Balanced softmax (Ren et
al. 2020) is the same construction at ``tau = 1``.

Both are training-time only. At eval the raw logits are already the balanced
scores, so nothing downstream (argmax, mIoU, uncertainty, neural PCA) changes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_CHOICES: tuple[str, ...] = ("ce", "balanced_softmax", "logit_adjusted")

# Floor for empty/near-empty classes: log(0) would be -inf and poison the logits.
_PRIOR_EPS = 1e-12


class LogitAdjustedCrossEntropy(nn.Module):
    """Cross-entropy on logits shifted by ``tau * log(prior)``.

    With uniform priors the shift is the same constant on every class, and
    softmax is shift-invariant, so this reduces exactly to plain cross-entropy.
    """

    def __init__(
        self,
        priors: torch.Tensor,
        *,
        tau: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        if priors.ndim != 1:
            raise ValueError(f"priors must be 1-D [num_classes], got {tuple(priors.shape)}")
        priors = priors.detach().to(dtype=torch.float32)
        total = float(priors.sum())
        if total <= 0:
            raise ValueError("priors sum to zero -- no labeled pixels were counted")
        priors = priors / total
        self.tau = float(tau)
        self.ignore_index = int(ignore_index)
        # Buffer (not parameter) so it rides device moves and lands in state_dict.
        self.register_buffer(
            "adjustment", self.tau * torch.log(priors.clamp_min(_PRIOR_EPS))
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits [B, C, H, W]; broadcast the per-class shift over B, H, W
        shift = self.adjustment.view(1, -1, *([1] * (logits.ndim - 2)))
        return F.cross_entropy(
            logits + shift.to(logits.dtype), target, ignore_index=self.ignore_index
        )


def build_segmentation_loss(
    loss_name: str,
    *,
    num_classes: int,
    ignore_index: int,
    priors: Optional[torch.Tensor] = None,
    tau: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Construct the Stage-3 loss.

    ``ce`` is the historical default and is returned unchanged so existing runs
    reproduce exactly. The long-tail variants need ``priors``; passing
    ``class_weights`` alongside them is refused rather than silently stacked --
    logit adjustment already corrects the prior, so doing both double-counts it.
    """
    if loss_name not in LOSS_CHOICES:
        raise ValueError(f"Unknown loss {loss_name!r}; expected one of {LOSS_CHOICES}")

    if loss_name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)

    if priors is None:
        raise ValueError(f"loss={loss_name!r} requires class priors")
    if priors.numel() != num_classes:
        raise ValueError(
            f"priors has {priors.numel()} entries but num_classes={num_classes}"
        )
    if class_weights is not None:
        raise ValueError(
            f"loss={loss_name!r} cannot be combined with class weights -- logit "
            f"adjustment already corrects for the class prior, so applying both "
            f"double-counts the imbalance. Drop the class-weight vector."
        )

    # Balanced softmax is logit adjustment pinned at tau=1.
    effective_tau = 1.0 if loss_name == "balanced_softmax" else float(tau)
    return LogitAdjustedCrossEntropy(
        priors, tau=effective_tau, ignore_index=ignore_index
    )


@torch.no_grad()
def compute_pixel_class_priors(
    dataloader,
    *,
    num_classes: int,
    ignore_index: int,
    max_batches: Optional[int] = None,
    parse_batch=None,
    remap: Optional[Sequence[int]] = None,
    device=None,
) -> torch.Tensor:
    """Per-class pixel frequency over a loader's targets, excluding ``ignore_index``.

    Returns counts (not normalized) as float64 so downstream normalization is
    exact; ``build_segmentation_loss`` normalizes. ``remap`` optionally projects
    DC target ids through a lookup (used to derive IG priors from DC labels)
    before counting.
    """
    if parse_batch is None:
        from vision_backend.training.utils import parse_segmentation_batch as parse_batch

    counts = torch.zeros(num_classes, dtype=torch.float64)
    seen = 0
    for index, batch in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break
        _, target, _ = parse_batch(batch)
        target = target.long()
        if remap is not None:
            from vision_backend.training.hierarchy import map_dc_targets_to_ig

            lookup = torch.as_tensor(list(remap), dtype=torch.long)
            target = map_dc_targets_to_ig(target, lookup, ignore_index)
        valid = target[target != ignore_index]
        if valid.numel():
            counts += torch.bincount(
                valid.reshape(-1).cpu(), minlength=num_classes
            ).to(torch.float64)
            seen += int(valid.numel())

    if seen == 0:
        raise ValueError(
            "No labeled pixels found while computing class priors -- every pixel "
            "was ignore_index. Check the manifest and label raster."
        )
    return counts
