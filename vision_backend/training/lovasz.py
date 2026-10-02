"""Lovász-softmax loss: a convex surrogate of per-class IoU.

Berman, Rannen Triki, Blaschko. "The Lovász-Softmax loss: A tractable surrogate
for the optimization of the intersection-over-union measure in neural
networks." CVPR 2018. Matches the authors' reference
(bermanmaxim/LovaszSoftmax, pytorch/lovasz_losses.py: `lovasz_softmax` with
classes='present', per_image=False): over all valid pixels of the batch, for
each class present in the ground truth, the pixel errors |1[y=c] - p_c| are
sorted in decreasing order and dotted with the gradient of the Lovász extension
of the Jaccard loss; the loss is the mean over those classes. For one-hot
predictions it equals 1 - mean IoU over the present classes exactly.

Every class weighs the same, whatever its pixel count -- the property mIoU has
and frequency-weighted cross-entropy lacks. All classes are sorted in one
row-wise sort over a [C, N] tensor instead of the reference's per-class loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lovasz_softmax(logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 255) -> torch.Tensor:
    """logits [B, C, H, W] (any normalisation; softmax is applied), target [B, H, W]."""
    c = logits.shape[1]
    probas = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, c)
    labels = target.reshape(-1)
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    if labels.numel() == 0:
        return logits.sum() * 0.0

    # [C, N], each class a contiguous row: sorting along the last, contiguous dim
    # takes ~14 ms for 1M pixels x 14 classes on an A100; the same sort along
    # dim 0 of an [N, C] tensor hits a slow path and took ~790 ms (10x a step).
    fg = F.one_hot(labels.long(), c).float().t().contiguous()
    errors = (fg - probas.t()).abs()
    errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
    fg_sorted = fg.gather(1, perm)
    gts = fg_sorted.sum(dim=1, keepdim=True)                    # pixels per class
    intersection = gts - fg_sorted.cumsum(dim=1)
    union = gts + (1.0 - fg_sorted).cumsum(dim=1)
    jaccard = 1.0 - intersection / union
    grad = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], dim=1)  # Lovász extension gradient
    per_class = (errors_sorted * grad).sum(dim=1)
    present = gts.squeeze(1) > 0
    return per_class[present].mean()
