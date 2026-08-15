"""Tests for the shared pre-classifier feature hook used by uncertainty + neural PCA."""

import pytest
import torch

from vision_backend.model.features import (
    extract_pixel_features,
    extract_pooled_features,
    get_classifier_head,
    get_classifier_weight_vector,
)
from vision_backend.training.builders import build_simmim_segmentation_model


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


def test_get_classifier_head_returns_conv2d(tiny_model):
    head = get_classifier_head(tiny_model)
    assert isinstance(head, torch.nn.Conv2d)
    assert head.out_channels == 3
    assert head.kernel_size == (1, 1)


def test_get_classifier_head_rejects_non_segmentation_model():
    with pytest.raises(ValueError, match="decoder.head"):
        get_classifier_head(torch.nn.Linear(4, 4))


def test_get_classifier_weight_vector_shape_and_index(tiny_model):
    head = get_classifier_head(tiny_model)
    w0 = get_classifier_weight_vector(tiny_model, 0)
    assert w0.shape == (head.in_channels,)
    assert torch.equal(w0, head.weight[0, :, 0, 0])


def test_get_classifier_weight_vector_out_of_range_raises(tiny_model):
    head = get_classifier_head(tiny_model)
    with pytest.raises(ValueError, match="out of range"):
        get_classifier_weight_vector(tiny_model, head.out_channels)


def test_extract_pixel_features_matches_input_resolution(tiny_model):
    x = torch.randn(2, 1, 256, 256)
    feats = extract_pixel_features(tiny_model, x)
    head = get_classifier_head(tiny_model)
    assert feats.shape == (2, head.in_channels, 256, 256)
    assert torch.isfinite(feats).all()


def test_extract_pooled_features_is_spatial_mean(tiny_model):
    x = torch.randn(1, 1, 256, 256)
    pixel_feats = extract_pixel_features(tiny_model, x)
    pooled = extract_pooled_features(tiny_model, x)
    assert pooled.shape == (1, pixel_feats.shape[1])
    assert torch.allclose(pooled, pixel_feats.mean(dim=(2, 3)), atol=1e-5)


def test_hook_is_removed_after_use(tiny_model):
    """Two calls in a row shouldn't accumulate hooks (would silently corrupt features)."""
    head = get_classifier_head(tiny_model)
    n_before = len(head._forward_pre_hooks)
    extract_pixel_features(tiny_model, torch.randn(1, 1, 256, 256))
    extract_pixel_features(tiny_model, torch.randn(1, 1, 256, 256))
    assert len(head._forward_pre_hooks) == n_before
