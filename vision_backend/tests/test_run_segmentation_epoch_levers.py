"""Tests for run_segmentation_epoch's F1 (IG aux loss), F2 (balanced-softmax /
logit-adjusted loss), and F5 (EMA) levers, and their "neutral default"
guarantee (every new kwarg off/None reproduces today's plain DC-CE path)."""

import pytest
import torch
import torch.nn.functional as F

from vision_backend.training.builders import build_simmim_segmentation_model
from vision_backend.training.ema import ModelEMA
from vision_backend.training.hierarchy import build_dc_to_ig_tensor
from vision_backend.training.utils import (
    adjust_logits_for_prior,
    compute_log_class_priors,
    run_segmentation_epoch,
)

IGNORE_INDEX = 255
SIMMIM_CONFIG = {
    "model_kind": "simmim",
    "in_channels": 1,
    "global_base_grid": 4,
    "window_size": 8,
    "decoder_channels": 8,
    "num_classes": 3,
}


def _model(**overrides):
    torch.manual_seed(0)
    return build_simmim_segmentation_model({**SIMMIM_CONFIG, **overrides})


def _dataloader(num_classes=3, n_batches=2, batch_size=1, size=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        image = torch.rand(batch_size, 1, size, size, generator=g) * 2 - 1
        label = torch.randint(0, num_classes, (batch_size, size, size), generator=g)
        batches.append({"image": image, "label": label})
    return batches


# --------------------------------------------------------------------------
# Neutral-default regression: every new kwarg off reproduces today's 3-key dict
# --------------------------------------------------------------------------
def test_defaults_return_exactly_the_original_three_keys():
    model = _model()
    metrics = run_segmentation_epoch(
        torch, model, _dataloader(), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
    )
    assert set(metrics.keys()) == {"loss", "pixel_acc", "miou"}


# --------------------------------------------------------------------------
# F1: IG aux head loss
# --------------------------------------------------------------------------
def test_ig_loss_weight_positive_without_dc_to_ig_raises():
    model = _model(num_classes_ig=2)
    with pytest.raises(ValueError, match="dc_to_ig"):
        run_segmentation_epoch(
            torch, model, _dataloader(), torch.device("cpu"),
            num_classes=3, ignore_index=IGNORE_INDEX, ig_loss_weight=0.4,
        )


def test_ig_loss_weight_positive_but_model_has_no_ig_head_raises_at_forward():
    model = _model()  # no num_classes_ig -> decoder.head_ig is None
    dc_to_ig = torch.tensor([0, 0, 1])
    with pytest.raises(RuntimeError, match="no IG head"):
        run_segmentation_epoch(
            torch, model, _dataloader(), torch.device("cpu"),
            num_classes=3, ignore_index=IGNORE_INDEX,
            ig_loss_weight=0.4, dc_to_ig=dc_to_ig, num_classes_ig=2,
        )


def test_ig_loss_weight_positive_with_matching_ig_head_reports_both_mious():
    model = _model(num_classes_ig=2)
    dc_to_ig = build_dc_to_ig_tensor(torch, 3, mapping={1: 1, 2: 1, 3: 2})
    metrics = run_segmentation_epoch(
        torch, model, _dataloader(), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        ig_loss_weight=0.4, dc_to_ig=dc_to_ig, num_classes_ig=2,
    )
    assert set(metrics.keys()) == {"loss", "pixel_acc", "miou", "pixel_acc_ig", "miou_ig"}
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_ig_loss_actually_receives_gradient_signal():
    """head_ig's weights must move when ig_loss_weight > 0 and training."""
    model = _model(num_classes_ig=2)
    dc_to_ig = build_dc_to_ig_tensor(torch, 3, mapping={1: 1, 2: 1, 3: 2})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.decoder.head_ig.weight.detach().clone()
    run_segmentation_epoch(
        torch, model, _dataloader(n_batches=1), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        optimizer=optimizer, ig_loss_weight=0.4, dc_to_ig=dc_to_ig, num_classes_ig=2,
    )
    after = model.decoder.head_ig.weight.detach()
    assert not torch.equal(before, after)


# --------------------------------------------------------------------------
# F2: balanced-softmax / logit-adjusted loss
# --------------------------------------------------------------------------
def test_uniform_priors_balanced_softmax_matches_plain_ce():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    log_priors = compute_log_class_priors(torch, {0: 10, 1: 10, 2: 10, 3: 10}, 4)
    adjusted = adjust_logits_for_prior(logits, log_priors, tau=1.0)
    ce_plain = F.cross_entropy(logits, target)
    ce_adjusted = F.cross_entropy(adjusted, target)
    assert ce_adjusted.item() == pytest.approx(ce_plain.item(), abs=1e-5)


def test_compute_log_class_priors_matches_expected_log_frequencies():
    counts = {0: 90, 1: 10}
    log_priors = compute_log_class_priors(torch, counts, 2)
    assert log_priors[0].item() == pytest.approx(torch.log(torch.tensor(0.9)).item(), abs=1e-5)
    assert log_priors[1].item() == pytest.approx(torch.log(torch.tensor(0.1)).item(), abs=1e-5)


def test_compute_log_class_priors_zero_count_class_is_finite_not_inf():
    log_priors = compute_log_class_priors(torch, {0: 100, 1: 0}, 2)
    assert torch.isfinite(log_priors[1])
    assert log_priors[1] < log_priors[0]


def test_run_segmentation_epoch_balanced_softmax_runs_with_finite_gradients():
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dc_log_priors = compute_log_class_priors(torch, {0: 50, 1: 30, 2: 20}, 3)
    metrics = run_segmentation_epoch(
        torch, model, _dataloader(n_batches=1), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        optimizer=optimizer, loss_kind="balanced_softmax", dc_log_priors=dc_log_priors,
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_run_segmentation_epoch_logit_adjusted_without_priors_raises():
    model = _model()
    with pytest.raises(ValueError, match="dc_log_priors"):
        run_segmentation_epoch(
            torch, model, _dataloader(n_batches=1), torch.device("cpu"),
            num_classes=3, ignore_index=IGNORE_INDEX, loss_kind="logit_adjusted",
        )


def test_balanced_softmax_does_not_stack_with_class_weights(capsys):
    model = _model()
    class_weights = torch.tensor([5.0, 1.0, 1.0])
    dc_log_priors = compute_log_class_priors(torch, {0: 50, 1: 30, 2: 20}, 3)
    run_segmentation_epoch(
        torch, model, _dataloader(n_batches=1), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        loss_kind="balanced_softmax", dc_log_priors=dc_log_priors, class_weights=class_weights,
    )
    captured = capsys.readouterr()
    assert "does not" in captured.out and "class_weights" in captured.out


# --------------------------------------------------------------------------
# F5: EMA
# --------------------------------------------------------------------------
def test_ema_updates_once_per_optimizer_step():
    model = _model()
    ema = ModelEMA(model, decay=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = {n: p.clone() for n, p in ema.ema.named_parameters()}
    run_segmentation_epoch(
        torch, model, _dataloader(n_batches=2), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        optimizer=optimizer, ema=ema,
    )
    after = dict(ema.ema.named_parameters())
    moved = any(not torch.equal(before[n], after[n]) for n in before)
    assert moved


def test_ema_not_updated_during_val_pass():
    model = _model()
    ema = ModelEMA(model, decay=0.5)
    before = {n: p.clone() for n, p in ema.ema.named_parameters()}
    run_segmentation_epoch(
        torch, model, _dataloader(n_batches=2), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        optimizer=None, ema=ema,  # val pass: no optimizer -> no steps -> no EMA update
    )
    after = dict(ema.ema.named_parameters())
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_ema_uses_ema_source_model_when_given():
    """Simulates the torch.compile case: `model` passed to run_segmentation_epoch
    differs from the plain module the EMA should track (ema_source_model)."""
    base_model = _model()
    ema = ModelEMA(base_model, decay=0.5)
    optimizer = torch.optim.SGD(base_model.parameters(), lr=0.5)
    before = {n: p.clone() for n, p in ema.ema.named_parameters()}
    run_segmentation_epoch(
        torch, base_model, _dataloader(n_batches=1), torch.device("cpu"),
        num_classes=3, ignore_index=IGNORE_INDEX,
        optimizer=optimizer, ema=ema, ema_source_model=base_model,
    )
    after = dict(ema.ema.named_parameters())
    assert any(not torch.equal(before[n], after[n]) for n in before)
