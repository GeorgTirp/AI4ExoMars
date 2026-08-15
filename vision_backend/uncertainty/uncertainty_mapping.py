"""Uncertainty maps: softmax-based (always available) and Mahalanobis-based (needs
fitted `MahalanobisStats`, see `malahanobis.py` + `fit_gaussians.py`).

This is the module callers outside AI4ExoMars (e.g. MarsObsLabeling's
`mars-inference`) should import from -- it wraps the lower-level pieces in
`malahanobis.py` and `model/features.py` into single-call functions that take a
model and raw input(s) and return a display-ready [0, 1] map.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from vision_backend.model.features import extract_pixel_features
except ModuleNotFoundError:
    from model.features import extract_pixel_features

from .malahanobis import MahalanobisStats, mahalanobis_distance_map


@torch.no_grad()
def softmax_confidence_map(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel max-softmax confidence: max_c p(c). Shape [B,C,H,W] -> [B,H,W].

    Higher = more confident. Needs no calibration -- always available for any
    trained (or even untrained) model, unlike the Mahalanobis map below.
    """
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values


@torch.no_grad()
def predictive_entropy_map(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel predictive entropy H(p). Shape [B,C,H,W] -> [B,H,W]. Higher = less certain."""
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs.clamp_min(1e-8))
    return -(probs * log_probs).sum(dim=1)


def epistemic_uncertainty_map(
    model,
    *inputs: torch.Tensor,
    stats: MahalanobisStats,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-pixel epistemic (OOD) uncertainty from the fitted Mahalanobis Gaussians.

    Parameters
    ----------
    model : nn.Module
        A SingleBranchSegmentationModel or ContextAwareSegmentationModel.
    *inputs : torch.Tensor
        Forwarded to the model (one tensor, or local+context for the context model).
    stats : MahalanobisStats
        Fitted via `malahanobis.fit_class_gaussians` (see `fit_gaussians.py`).
    normalize : bool
        If True (default), scale by `stats.reference_max_distance` and clamp to
        [0, 1] for direct use as a display heatmap. If False, return raw distances.

    Returns
    -------
    torch.Tensor
        [B, H, W], in [0, 1] if normalize else raw Mahalanobis distance.
    """
    pixel_features = extract_pixel_features(model, *inputs)
    distances = mahalanobis_distance_map(pixel_features, stats)
    if not normalize:
        return distances
    ref = stats.reference_max_distance or 1.0
    return (distances / max(ref, 1e-8)).clamp(0.0, 1.0)
