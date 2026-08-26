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


def _same_device(a, b) -> bool:
    """Device equality that treats an unindexed device as matching index 0.

    `torch.device("mps") != torch.device("mps:0")` even though a tensor placed on
    the former reports the latter, so a naive `==` would copy on every call.
    """
    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


@dataclass
class ClassGaussianStats:
    """Fitted Gaussian for one class's feature distribution."""

    mean: torch.Tensor  # [F]
    precision: torch.Tensor  # [F, F], (regularized covariance)^-1

    def to(self, device) -> "ClassGaussianStats":
        """Copy onto `device` (no-op if already there)."""
        if _same_device(self.mean.device, device):
            return self
        return ClassGaussianStats(
            mean=self.mean.to(device), precision=self.precision.to(device)
        )


@dataclass
class MahalanobisStats:
    """Per-class Gaussians plus enough metadata to score new features consistently."""

    class_stats: dict[int, ClassGaussianStats]
    feature_dim: int
    # A reference distance (e.g. a high percentile of the fitting set's own
    # in-distribution scores) used to normalize raw distances to ~[0, 1] for
    # display. None until fit_class_gaussians sets it.
    reference_max_distance: float | None = None

    @property
    def device(self):
        """Device the fitted tensors currently live on (None if there are no classes)."""
        for class_stat in self.class_stats.values():
            return class_stat.mean.device
        return None

    def to(self, device) -> "MahalanobisStats":
        """Copy every fitted tensor onto `device` (no-op if already there).

        `load_stats` always deserializes to CPU, so scoring features from a model
        on cuda/mps needs this first -- otherwise the subtraction in
        `_min_class_distance` raises "expected all tensors to be on the same device".
        """
        if self.device is not None and _same_device(self.device, device):
            return self
        return MahalanobisStats(
            class_stats={c: s.to(device) for c, s in self.class_stats.items()},
            feature_dim=self.feature_dim,
            reference_max_distance=self.reference_max_distance,
        )


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


# Rows scored per chunk in the shared-covariance path. Bounds the [chunk, F]
# temporaries: a full 4x512x512 feature map is ~1M rows, and materializing
# x @ precision for all of them at once costs ~1 GB at F=256.
_SCORE_CHUNK_ROWS = 1 << 16


def _has_shared_precision(class_stats: list[ClassGaussianStats]) -> bool:
    """True when every class shares one precision matrix (the default fit).

    `fit_class_gaussians(shared_covariance=True)` stores the *same* tensor object
    on every class, and torch.save/load preserves that aliasing, so the identity
    check nearly always settles it without touching the data.
    """
    first = class_stats[0].precision
    if all(cs.precision is first for cs in class_stats[1:]):
        return True
    return all(torch.equal(cs.precision, first) for cs in class_stats[1:])


def _min_distance_shared(
    features: torch.Tensor, class_stats: list[ClassGaussianStats], precision: torch.Tensor
) -> torch.Tensor:
    """Nearest-class Mahalanobis distance when all classes share one precision.

    Expanding d_c^2 = (x - u_c)' P (x - u_c) = x'Px - 2x'Pu_c + u_c'Pu_c lets the
    only expensive term (x'Px, an [N,F]x[F,F] matmul) be computed once for all
    classes instead of once per class. The cross term is [N,F]x[F,C], which is
    ~5% of that at C=13, F=256 -- so this is close to a C-fold reduction in work,
    and it never materializes a per-class [N,F] difference tensor.
    """
    means = torch.stack([cs.mean for cs in class_stats])  # [C, F]
    means_p = means @ precision  # [C, F]  (P is symmetric)
    const = (means_p * means).sum(dim=1)  # [C]  u_c'Pu_c

    out = []
    for chunk in features.split(_SCORE_CHUNK_ROWS, dim=0):
        chunk_p = chunk @ precision  # [n, F]  x'P
        quad = (chunk_p * chunk).sum(dim=1)  # [n]    x'Px
        cross = chunk_p @ means.t()  # [n, C]  x'Pu_c
        d2 = quad.unsqueeze(1) - 2.0 * cross + const.unsqueeze(0)  # [n, C]
        out.append(d2.min(dim=1).values.clamp_min(0.0).sqrt())
    return torch.cat(out, dim=0)


def _min_class_distance(features: torch.Tensor, stats: MahalanobisStats) -> torch.Tensor:
    """Per-sample Mahalanobis distance to the nearest class Gaussian. features: [N, F] -> [N]."""
    # Stats come off disk on CPU while features follow the model (cuda/mps).
    # Callers scoring in bulk should hoist `stats.to(device)` out of their loop;
    # this keeps one-off callers correct rather than raising a device mismatch.
    stats = stats.to(features.device)
    class_stats = list(stats.class_stats.values())
    if not class_stats:
        raise ValueError("MahalanobisStats has no fitted classes to score against")

    if _has_shared_precision(class_stats):
        return _min_distance_shared(features, class_stats, class_stats[0].precision)

    # Per-class covariance: no shared term to hoist, score one class at a time.
    distances = []
    for class_stat in class_stats:
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
