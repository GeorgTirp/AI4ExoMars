"""Tests for training.utils.focal_loss."""

import pytest
import torch
import torch.nn.functional as F

from vision_backend.training.utils import focal_loss

IGNORE_INDEX = 255


def _random_logits_and_target(seed=0, b=2, c=5, h=4, w=4):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, c, h, w, generator=g)
    target = torch.randint(0, c, (b, h, w), generator=g)
    return logits, target


def test_gamma_zero_matches_plain_cross_entropy_exactly():
    """(1-pt)^0 == 1 for all pixels -> focal reduces to plain CE exactly."""
    logits, target = _random_logits_and_target()
    focal = focal_loss(torch, logits, target, gamma=0.0, ignore_index=IGNORE_INDEX)
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    assert focal.item() == pytest.approx(ce.item(), abs=1e-5)


def test_gamma_zero_with_weight_matches_weighted_cross_entropy_exactly():
    """Same reduction convention as nn.CrossEntropyLoss(weight=...): mean =
    sum(weight[t]*loss)/sum(weight[t]), not a plain per-pixel mean."""
    logits, target = _random_logits_and_target(c=5)
    weight = torch.tensor([0.1, 2.0, 1.0, 0.5, 3.0])
    focal = focal_loss(torch, logits, target, weight=weight, gamma=0.0, ignore_index=IGNORE_INDEX)
    ce = F.cross_entropy(logits, target, weight=weight, ignore_index=IGNORE_INDEX)
    assert focal.item() == pytest.approx(ce.item(), abs=1e-5)


def test_ignore_index_excluded_from_loss():
    logits, target = _random_logits_and_target(c=3, h=2, w=2)
    target_with_ignore = target.clone()
    target_with_ignore[0, 0, 0] = IGNORE_INDEX

    # Loss over the reduced (non-ignored) set should match computing focal loss
    # on a target with that pixel's class value removed from consideration --
    # verified indirectly: gamma=0 case must still match F.cross_entropy, which
    # has its own correct ignore_index handling.
    focal = focal_loss(torch, logits, target_with_ignore, gamma=0.0, ignore_index=IGNORE_INDEX)
    ce = F.cross_entropy(logits, target_with_ignore, ignore_index=IGNORE_INDEX)
    assert focal.item() == pytest.approx(ce.item(), abs=1e-5)


def test_all_ignored_does_not_nan_or_crash():
    logits, target = _random_logits_and_target(c=3, h=2, w=2)
    target[:] = IGNORE_INDEX
    loss = focal_loss(torch, logits, target, gamma=2.0, ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss)


def test_higher_gamma_downweights_confident_correct_pixel_more():
    """A pixel the model is already very confident (and correct) about should
    contribute less to the loss as gamma increases; the gradient through it
    should shrink accordingly."""
    torch.manual_seed(0)
    num_classes = 3
    logits = torch.zeros(1, num_classes, 1, 1, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0, 0] = 10.0  # model is very confident about class 0
    target = torch.tensor([[[0]]])  # and it's correct

    losses = {}
    for gamma in (0.0, 2.0, 5.0):
        logits_g = logits.detach().clone().requires_grad_(True)
        loss = focal_loss(torch, logits_g, target, gamma=gamma, ignore_index=IGNORE_INDEX)
        loss.backward()
        losses[gamma] = (loss.item(), logits_g.grad.abs().sum().item())

    assert losses[2.0][0] < losses[0.0][0]
    assert losses[5.0][0] < losses[2.0][0]
    assert losses[2.0][1] < losses[0.0][1]
    assert losses[5.0][1] < losses[2.0][1]


def test_focal_loss_runs_on_segmentation_shaped_batch_no_nan():
    logits, target = _random_logits_and_target(b=4, c=14, h=64, w=64)
    weight = torch.rand(14) + 0.1
    loss = focal_loss(torch, logits, target, weight=weight, gamma=2.0, ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_unweighted_denominator_is_valid_pixel_count():
    """Without a weight tensor, the mean should divide by the count of
    non-ignored pixels (matches plain CrossEntropyLoss(reduction='mean'))."""
    logits, target = _random_logits_and_target(c=4, h=3, w=3)
    target[0, 0, 0] = IGNORE_INDEX  # one ignored pixel
    focal = focal_loss(torch, logits, target, gamma=0.0, ignore_index=IGNORE_INDEX)
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="mean")
    assert focal.item() == pytest.approx(ce.item(), abs=1e-5)
