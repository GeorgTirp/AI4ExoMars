"""Tests for training.utils.compute_class_weights (class-imbalance mitigation)."""

import pytest
import torch

from vision_backend.training.utils import compute_class_weights


def test_rare_class_gets_higher_weight_than_common_class():
    counts = {0: 10000, 1: 100, 2: 5000}
    weights = compute_class_weights(torch, counts, num_classes=3, scheme="inverse_sqrt")
    assert weights[1] > weights[2] > weights[0]


def test_zero_count_class_gets_zero_weight_not_infinite():
    counts = {0: 1000, 1: 0, 2: 500}
    weights = compute_class_weights(torch, counts, num_classes=3, scheme="inverse_sqrt")
    assert weights[1] == 0.0
    assert torch.isfinite(weights).all()


def test_present_class_weights_normalize_to_mean_one():
    counts = {0: 1000, 1: 50, 2: 500, 3: 0}
    weights = compute_class_weights(torch, counts, num_classes=4, scheme="inverse_sqrt", clip_max=None)
    present = weights[weights > 0]
    assert present.mean().item() == pytest.approx(1.0, abs=1e-4)


def test_clip_max_bounds_weight_ratio():
    counts = {0: 1_000_000, 1: 1}
    unclipped = compute_class_weights(torch, counts, num_classes=2, scheme="inverse_sqrt", clip_max=None)
    clipped = compute_class_weights(torch, counts, num_classes=2, scheme="inverse_sqrt", clip_max=10.0)

    assert unclipped[1] / unclipped[0] > 100  # would be extreme unclipped
    assert clipped[1] / clipped[0] <= 10.0 + 1e-6


def test_inverse_scheme_more_aggressive_than_inverse_sqrt():
    counts = {0: 10000, 1: 100}
    sqrt_weights = compute_class_weights(torch, counts, num_classes=2, scheme="inverse_sqrt", clip_max=None)
    inv_weights = compute_class_weights(torch, counts, num_classes=2, scheme="inverse", clip_max=None)

    sqrt_ratio = (sqrt_weights[1] / sqrt_weights[0]).item()
    inv_ratio = (inv_weights[1] / inv_weights[0]).item()
    assert inv_ratio > sqrt_ratio


def test_effective_number_scheme_runs_and_upweights_rare_class():
    counts = {0: 10000, 1: 100, 2: 5000}
    weights = compute_class_weights(torch, counts, num_classes=3, scheme="effective_number")
    # The rare class (100) is upweighted vs both common ones. Classes far above
    # the beta scale (5000, 10000) saturate 1-beta^n -> 1 and land on the same
    # weight -- that's the formula behaving as designed (Cui et al. 2019), not
    # a bug: it only differentiates near the rarest class's own scale.
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    assert torch.isfinite(weights).all()


def test_accepts_list_input_as_well_as_dict():
    counts_list = [10000, 100, 5000]
    counts_dict = {0: 10000, 1: 100, 2: 5000}
    w_list = compute_class_weights(torch, counts_list, num_classes=3, scheme="inverse_sqrt")
    w_dict = compute_class_weights(torch, counts_dict, num_classes=3, scheme="inverse_sqrt")
    assert torch.allclose(w_list, w_dict)


def test_list_input_wrong_length_raises():
    with pytest.raises(ValueError, match="Expected 3 counts"):
        compute_class_weights(torch, [1, 2], num_classes=3, scheme="inverse_sqrt")


def test_all_zero_counts_raises():
    with pytest.raises(ValueError, match="zero"):
        compute_class_weights(torch, {0: 0, 1: 0}, num_classes=2, scheme="inverse_sqrt")


def test_missing_dict_keys_default_to_zero_count():
    # class 2 absent from the dict entirely -- same as explicit 0.
    counts = {0: 1000, 1: 500}
    weights = compute_class_weights(torch, counts, num_classes=3, scheme="inverse_sqrt")
    assert weights[2] == 0.0


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown scheme"):
        compute_class_weights(torch, {0: 10, 1: 10}, num_classes=2, scheme="bogus")


def test_realistic_imbalance_shape_matches_this_project():
    """Sanity check against the AOI's actual (highly skewed) class distribution."""
    counts = {
        0: 5019, 1: 5040, 2: 5000, 3: 1142, 4: 5013, 5: 5039, 6: 4736,
        7: 5011, 8: 5011, 9: 2493, 10: 5000, 11: 3503, 12: 5008, 13: 0,
    }
    weights = compute_class_weights(torch, counts, num_classes=14, scheme="inverse_sqrt")
    assert weights[13] == 0.0  # boulder fields: no samples here, can't be weighted
    assert weights[3] > weights[0]  # rarer "Smooth bedrock" upweighted vs common class
    assert torch.isfinite(weights).all()
