"""Epoch mIoU must be global, not a mean of per-batch mIoUs.

IoU is a ratio of pixel counts, so averaging it over batches is not the
dataset value. Worse, each batch averages over only the classes present in it,
which reweights rare classes by how often they co-occur with others. That
number chose the best epoch's checkpoint, the best trial per variant, and the
v0-v3 ranking, so the difference is not cosmetic.

These tests pin the global definition and prove the two disagree.
"""

from __future__ import annotations

import torch

from vision_backend.training.utils import (
    _accumulate_confusion,
    _compute_segmentation_metrics,
    _metrics_from_confusion,
)

NUM_CLASSES = 3
IGNORE = 255


def _logits_for(pred: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """One-hot logits whose argmax is exactly `pred` (shape [B, H, W])."""
    return torch.nn.functional.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()


def test_confusion_matches_single_batch_metrics_when_there_is_one_batch():
    """With a single batch the two definitions must agree exactly."""
    target = torch.tensor([[[0, 0, 1, 2]]])
    pred = torch.tensor([[[0, 1, 1, 2]]])
    logits = _logits_for(pred)

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    _accumulate_confusion(torch, conf, logits, target,
                          num_classes=NUM_CLASSES, ignore_index=IGNORE)
    globally = _metrics_from_confusion(torch, conf)
    per_batch = _compute_segmentation_metrics(
        torch, logits, target, num_classes=NUM_CLASSES, ignore_index=IGNORE
    )
    assert globally["miou"] == per_batch["miou"]
    assert globally["pixel_acc"] == per_batch["pixel_acc"]


def test_global_miou_differs_from_the_mean_of_per_batch_mious():
    """The regression this guards: batch composition changing the score.

    Batch A contains only class 0 and gets it perfectly, so its per-batch mIoU
    is 1.0 -- it averages over the single class that happens to be present.
    Batch B contains only class 1 and half of it leaks into class 2, scoring
    0.25 over the two classes it sees. Their mean is 0.625.

    Globally the model is right on 3 of 4 pixels with class 2 never correct:
    IoUs are 1.0, 0.5 and 0.0, so the dataset mIoU is 0.5. The batch mean
    flatters the model by 0.125 purely because class 0 was alone in its batch.
    """
    a_target = torch.tensor([[[0, 0]]])
    a_pred = torch.tensor([[[0, 0]]])          # perfect, class 0 only
    b_target = torch.tensor([[[1, 1]]])
    b_pred = torch.tensor([[[1, 2]]])          # half leaks to class 2

    batches = [(a_pred, a_target), (b_pred, b_target)]

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    per_batch_mious = []
    for pred, target in batches:
        logits = _logits_for(pred)
        _accumulate_confusion(torch, conf, logits, target,
                              num_classes=NUM_CLASSES, ignore_index=IGNORE)
        per_batch_mious.append(
            _compute_segmentation_metrics(
                torch, logits, target,
                num_classes=NUM_CLASSES, ignore_index=IGNORE,
            )["miou"]
        )

    batch_mean = sum(per_batch_mious) / len(per_batch_mious)
    globally = _metrics_from_confusion(torch, conf)["miou"]

    # Global: class0 I=2 U=2 -> 1.0 ; class1 I=1 U=2 -> 0.5 ; class2 I=0 U=1 -> 0
    assert abs(globally - 0.5) < 1e-9
    assert abs(batch_mean - 0.625) < 1e-9
    assert abs(batch_mean - globally) > 0.1, (
        f"batch-mean {batch_mean} vs global {globally} -- if these agree the "
        "test case no longer exercises the difference"
    )


def test_ignore_index_is_excluded_from_both_numerator_and_denominator():
    target = torch.tensor([[[0, 1, IGNORE, IGNORE]]])
    pred = torch.tensor([[[0, 1, 2, 2]]])
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    _accumulate_confusion(torch, conf, _logits_for(pred), target,
                          num_classes=NUM_CLASSES, ignore_index=IGNORE)
    assert int(conf.sum()) == 2, "ignored pixels must not enter the matrix"
    assert _metrics_from_confusion(torch, conf)["pixel_acc"] == 1.0


def test_empty_split_is_zero_not_nan():
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    out = _metrics_from_confusion(torch, conf)
    assert out == {"pixel_acc": 0.0, "miou": 0.0}
