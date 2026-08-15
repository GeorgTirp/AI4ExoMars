"""Tests for Mahalanobis-distance epistemic uncertainty (uncertainty/malahanobis.py)."""

import pytest
import torch

from vision_backend.training.builders import build_simmim_segmentation_model
from vision_backend.uncertainty.malahanobis import (
    fit_class_gaussians,
    load_stats,
    mahalanobis_distance_map,
    save_stats,
)
from vision_backend.uncertainty.uncertainty_mapping import (
    epistemic_uncertainty_map,
    predictive_entropy_map,
    softmax_confidence_map,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = {
        "model_kind": "simmim",
        "in_channels": 1,
        "global_base_grid": 4,
        "window_size": 8,
        "decoder_channels": 16,
        "num_classes": 3,
    }
    return build_simmim_segmentation_model(config).eval()


def _synthetic_features(n_per_class=40, feature_dim=8, n_classes=3, sep=6.0, seed=0):
    """Well-separated per-class Gaussian clusters, so scoring assertions are meaningful."""
    g = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for c in range(n_classes):
        center = torch.zeros(feature_dim)
        center[c % feature_dim] = sep * c
        feats.append(center + torch.randn(n_per_class, feature_dim, generator=g))
        labels.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels)


def test_fit_class_gaussians_shared_covariance():
    features, labels = _synthetic_features()
    stats = fit_class_gaussians(features, labels, shared_covariance=True)

    assert set(stats.class_stats.keys()) == {0, 1, 2}
    assert stats.feature_dim == features.shape[1]
    for cs in stats.class_stats.values():
        assert cs.mean.shape == (features.shape[1],)
        assert cs.precision.shape == (features.shape[1], features.shape[1])
    assert stats.reference_max_distance is not None and stats.reference_max_distance > 0


def test_fit_class_gaussians_per_class_covariance_differs_from_shared():
    features, labels = _synthetic_features()
    shared = fit_class_gaussians(features, labels, shared_covariance=True)
    per_class = fit_class_gaussians(features, labels, shared_covariance=False)

    # Precision matrices should differ across classes when fit independently...
    assert not torch.allclose(per_class.class_stats[0].precision, per_class.class_stats[1].precision)
    # ...but are identical (the shared covariance) under shared_covariance=True.
    assert torch.allclose(shared.class_stats[0].precision, shared.class_stats[1].precision)


def test_in_distribution_sample_scores_lower_than_far_outlier():
    features, labels = _synthetic_features()
    stats = fit_class_gaussians(features, labels, shared_covariance=True)

    in_dist = features[labels == 0][0:1]  # a genuine class-0 sample
    outlier = torch.full((1, features.shape[1]), 1000.0)  # absurdly far from everything

    in_dist_map = mahalanobis_distance_map(in_dist.unsqueeze(-1).unsqueeze(-1), stats)
    outlier_map = mahalanobis_distance_map(outlier.unsqueeze(-1).unsqueeze(-1), stats)

    assert in_dist_map.item() < outlier_map.item()


def test_mahalanobis_distance_map_shape():
    features, labels = _synthetic_features(feature_dim=4)
    stats = fit_class_gaussians(features, labels)
    pixel_features = torch.randn(2, 4, 5, 6)  # [B, F, H, W]
    dist_map = mahalanobis_distance_map(pixel_features, stats)
    assert dist_map.shape == (2, 5, 6)
    assert torch.isfinite(dist_map).all()


def test_mahalanobis_distance_map_wrong_feature_dim_raises():
    features, labels = _synthetic_features(feature_dim=4)
    stats = fit_class_gaussians(features, labels)
    with pytest.raises(ValueError, match="channels"):
        mahalanobis_distance_map(torch.randn(1, 8, 3, 3), stats)


def test_save_load_stats_roundtrip(tmp_path):
    features, labels = _synthetic_features()
    stats = fit_class_gaussians(features, labels)
    path = tmp_path / "stats.pt"
    save_stats(stats, path)

    loaded = load_stats(path)
    assert loaded.feature_dim == stats.feature_dim
    assert set(loaded.class_stats.keys()) == set(stats.class_stats.keys())
    assert torch.allclose(loaded.class_stats[0].mean, stats.class_stats[0].mean)


def test_load_stats_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stats(tmp_path / "does_not_exist.pt")


def test_epistemic_uncertainty_map_end_to_end(tiny_model):
    """Fit on random features but exercise the real model's forward + hook path."""
    features, labels = _synthetic_features(feature_dim=16)  # matches decoder_channels=16
    stats = fit_class_gaussians(features, labels)

    x = torch.randn(1, 1, 256, 256)
    heat = epistemic_uncertainty_map(tiny_model, x, stats=stats, normalize=True)
    assert heat.shape == (1, 256, 256)
    assert (heat >= 0).all() and (heat <= 1).all()

    raw = epistemic_uncertainty_map(tiny_model, x, stats=stats, normalize=False)
    assert torch.isfinite(raw).all()


def test_softmax_confidence_and_entropy_maps(tiny_model):
    x = torch.randn(1, 1, 256, 256)
    logits = tiny_model(x)

    confidence = softmax_confidence_map(logits)
    entropy = predictive_entropy_map(logits)

    assert confidence.shape == (1, 256, 256)
    assert (confidence >= 0).all() and (confidence <= 1).all()
    assert entropy.shape == (1, 256, 256)
    assert (entropy >= 0).all()
