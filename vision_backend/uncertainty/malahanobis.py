"""Mahalanobis-distance out-of-distribution (OOD) / epistemic uncertainty.

Standard post-hoc OOD detection (Lee et al. 2018, "A Simple Unified Framework
for Detecting Out-of-Distribution Samples..."): fit a per-class Gaussian over a
trained model's penultimate-layer features on the training set, then at
inference time score each sample by its Mahalanobis distance to the *nearest*
class Gaussian. Large distance = far from anything the model was trained on =
high epistemic uncertainty / likely OOD.

This module only *scores* features -- fitting requires a full pass over a
labeled training set with a trained model (see `fit_gaussians.py`), and
scoring a single image requires a fitted `MahalanobisStats` produced by that
pass. Until a trained checkpoint + fitted stats exist, callers should expect
`load_stats` to simply not find a file yet -- that's expected, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from vision_backend.model.features import extract_pixel_features
except ModuleNotFoundError:
    from model.features import extract_pixel_features


@dataclass
class ClassGaussianStats:
    """Fitted Gaussian for one class's feature distribution."""

    mean: torch.Tensor  # [F]
    precision: torch.Tensor  # [F, F], (regularized covariance)^-1


@dataclass
class MahalanobisStats:
    """Per-class Gaussians plus enough metadata to score new features consistently."""

    class_stats: dict[int, ClassGaussianStats]
    feature_dim: int
    # A reference distance (e.g. a high percentile of the fitting set's own
    # in-distribution scores) used to normalize raw distances to ~[0, 1] for
    # display. None until fit_class_gaussians sets it.
    reference_max_distance: float | None = None


def fit_class_gaussians(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    shared_covariance: bool = True,
    eps: float = 1e-6,
) -> MahalanobisStats:
    """Fit per-class Gaussians (mean + precision) from labeled feature vectors.

    Parameters
    ----------
    features : torch.Tensor
        [N, F] feature vectors (e.g. pooled or per-pixel features flattened
        across many images), one row per sample.
    labels : torch.Tensor
        [N] integer class id per row.
    shared_covariance : bool
        If True (recommended, and the standard Lee et al. formulation), all
        classes share one covariance estimated from every sample's residual
        to its own class mean -- far more stable with limited data than a
        separate covariance per class. If False, each class gets its own.
    eps : float
        Ridge added to the covariance diagonal before inverting, for
        numerical stability when N is small relative to F.

    Returns
    -------
    MahalanobisStats
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, F], got shape {tuple(features.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same length")

    features = features.detach().float()
    labels = labels.detach().long()
    feature_dim = features.shape[1]
    class_ids = sorted(int(c) for c in torch.unique(labels).tolist())

    means: dict[int, torch.Tensor] = {}
    for c in class_ids:
        means[c] = features[labels == c].mean(dim=0)

    identity = torch.eye(feature_dim, dtype=features.dtype)

    if shared_covariance:
        centered = torch.cat(
            [features[labels == c] - means[c] for c in class_ids], dim=0
        )
        cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
        precision = torch.linalg.pinv(cov + eps * identity)
        class_stats = {c: ClassGaussianStats(mean=means[c], precision=precision) for c in class_ids}
    else:
        class_stats = {}
        for c in class_ids:
            centered = features[labels == c] - means[c]
            n = centered.shape[0]
            cov = (centered.T @ centered) / max(1, n - 1)
            precision = torch.linalg.pinv(cov + eps * identity)
            class_stats[c] = ClassGaussianStats(mean=means[c], precision=precision)

    stats = MahalanobisStats(class_stats=class_stats, feature_dim=feature_dim)
    # Calibrate a display reference scale from the fitting set's own in-distribution
    # scores (95th percentile), so downstream normalization has a sane default.
    in_dist_scores = _min_class_distance(features, stats)
    stats.reference_max_distance = float(torch.quantile(in_dist_scores, 0.95).item())
    return stats


def _min_class_distance(features: torch.Tensor, stats: MahalanobisStats) -> torch.Tensor:
    """Per-sample Mahalanobis distance to the nearest class Gaussian. features: [N, F] -> [N]."""
    distances = []
    for class_stat in stats.class_stats.values():
        diff = features - class_stat.mean  # [N, F]
        # d^2 = diff @ precision @ diff^T, computed row-wise without materializing [N,N]
        d2 = torch.einsum("nf,fg,ng->n", diff, class_stat.precision, diff)
        distances.append(d2.clamp_min(0.0).sqrt())
    return torch.stack(distances, dim=0).min(dim=0).values


def mahalanobis_distance_map(pixel_features: torch.Tensor, stats: MahalanobisStats) -> torch.Tensor:
    """Per-pixel distance to the nearest class Gaussian.

    Parameters
    ----------
    pixel_features : torch.Tensor
        [B, F, H, W] spatial feature map (see `model.features.extract_pixel_features`).
    stats : MahalanobisStats

    Returns
    -------
    torch.Tensor
        [B, H, W] Mahalanobis distance, higher = more out-of-distribution.
    """
    if pixel_features.shape[1] != stats.feature_dim:
        raise ValueError(
            f"pixel_features has {pixel_features.shape[1]} channels, "
            f"stats were fit on {stats.feature_dim}-dim features"
        )
    b, f, h, w = pixel_features.shape
    flat = pixel_features.permute(0, 2, 3, 1).reshape(-1, f)  # [B*H*W, F]
    dist = _min_class_distance(flat, stats)  # [B*H*W]
    return dist.reshape(b, h, w)


def epistemic_uncertainty_from_model(model, *inputs: torch.Tensor, stats: MahalanobisStats) -> torch.Tensor:
    """Convenience: run the model, then score its pre-classifier features.

    `*inputs` is forwarded to the model as in `model.features.extract_pixel_features`
    (one tensor for the single-branch model, two for the context-branch model).
    """
    pixel_features = extract_pixel_features(model, *inputs)
    return mahalanobis_distance_map(pixel_features, stats)


def save_stats(stats: MahalanobisStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)


def load_stats(path: str | Path) -> MahalanobisStats:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No fitted Mahalanobis stats at {path}")
    # weights_only=False: first-party artifact (dataclass of tensors), not a checkpoint.
    return torch.load(path, map_location="cpu", weights_only=False)
