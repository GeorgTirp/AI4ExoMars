"""Tests for training/lovasz.py (Lovász-softmax, Berman et al. CVPR 2018) and
its wiring into run_segmentation_epoch via lovasz_weight."""

import pytest
import torch
import torch.nn.functional as F

from vision_backend.training.builders import build_simmim_segmentation_model
from vision_backend.training.lovasz import lovasz_softmax
from vision_backend.training.utils import run_segmentation_epoch

IGNORE = 255


def _reference_lovasz(probas, labels):
    """Literal port of the authors' per-class loop (lovasz_softmax_flat,
    classes='present') on already-flattened [N, C] probabilities."""
    def lovasz_grad(gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    losses = []
    for c in range(probas.shape[1]):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


def test_matches_the_reference_per_class_implementation():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 16, 16, generator=g)
    target = torch.randint(0, 4, (2, 16, 16), generator=g)  # class 4 absent -> skipped
    ref = _reference_lovasz(F.softmax(logits, 1).permute(0, 2, 3, 1).reshape(-1, 5), target.reshape(-1))
    assert torch.allclose(lovasz_softmax(logits, target), ref, atol=1e-6)


def test_hard_predictions_give_one_minus_mean_iou_over_present_classes():
    g = torch.Generator().manual_seed(1)
    target = torch.randint(0, 4, (2, 20, 20), generator=g)
    pred = torch.randint(0, 5, (2, 20, 20), generator=g)
    logits = 60.0 * F.one_hot(pred, 5).permute(0, 3, 1, 2).float()
    ious = []
    for c in range(5):
        if (target == c).any():
            inter = ((pred == c) & (target == c)).sum()
            union = ((pred == c) | (target == c)).sum()
            ious.append(float(inter) / float(union))
    assert float(lovasz_softmax(logits, target)) == pytest.approx(1 - sum(ious) / len(ious), abs=1e-5)


def test_ignored_pixels_do_not_matter_and_perfect_prediction_is_zero():
    g = torch.Generator().manual_seed(2)
    target = torch.randint(0, 3, (1, 12, 12), generator=g)
    target[0, :4] = IGNORE
    logits = 50.0 * F.one_hot(target.clamp(max=2), 3).permute(0, 3, 1, 2).float()
    noisy = logits.clone()
    noisy[0, :, :4] = torch.randn(3, 4, 12, generator=g) * 10  # only ignored rows change
    assert float(lovasz_softmax(logits, target)) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(lovasz_softmax(noisy, target), lovasz_softmax(logits, target))


def test_gradient_is_finite_and_nonzero():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    lovasz_softmax(logits, torch.randint(0, 4, (2, 8, 8))).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_lovasz_weight_adds_the_lovasz_term_to_the_loss():
    def run(weight, training):
        torch.manual_seed(0)
        model = build_simmim_segmentation_model({
            "model_kind": "simmim", "in_channels": 1, "global_base_grid": 4,
            "window_size": 8, "decoder_channels": 8, "num_classes": 3,
        })
        g = torch.Generator().manual_seed(0)
        batches = [{"image": torch.rand(1, 1, 256, 256, generator=g),
                    "label": torch.randint(0, 3, (1, 256, 256), generator=g)}]
        opt = torch.optim.SGD(model.parameters(), lr=0.0) if training else None
        return run_segmentation_epoch(torch, model, batches, torch.device("cpu"), num_classes=3,
                                      ignore_index=IGNORE, optimizer=opt, lovasz_weight=weight)["loss"]

    # the Lovász term lies in [0, 1]; train_stage passes lovasz_weight to the
    # training epoch only, so the validation loss stays plain CE
    assert run(1.0, training=True) > run(0.0, training=True) + 0.1
    assert run(1.0, training=False) > run(0.0, training=False) + 0.1
