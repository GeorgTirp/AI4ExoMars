"""Shared feature-extraction hooks for post-hoc analysis (uncertainty, neural PCA).

Both `SingleBranchSegmentationModel` and `ContextAwareSegmentationModel`
(`training/segmentation.py`) share one structural landmark regardless of encoder:
a final ``decoder.head`` -- a plain ``nn.Conv2d(decoder_channels, num_classes, 1)``
applied to a feature map already resampled to the input's spatial resolution
(`LightweightSegmentationDecoder.forward` upsamples to `output_size` immediately
before calling `head`). Hooking that conv's *input* therefore gives, for any
current or future decoder internals, the same two things every downstream
analysis in `uncertainty/` and `pc_align/` needs:

- a per-pixel feature map already aligned with the input image (no extra
  upsampling bookkeeping), for spatial uncertainty maps;
- its global-average-pool, i.e. phi(x) in the neural-PCA method.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


def get_classifier_head(model: nn.Module) -> nn.Conv2d:
    """Return the model's final 1x1 classifier conv (`model.decoder.head`)."""
    decoder = getattr(model, "decoder", None)
    head = getattr(decoder, "head", None) if decoder is not None else None
    if not isinstance(head, nn.Conv2d):
        raise ValueError(
            "Model has no `decoder.head` Conv2d classifier -- expected a "
            "SingleBranchSegmentationModel or ContextAwareSegmentationModel "
            "(training/segmentation.py)."
        )
    return head


def get_classifier_weight_vector(model: nn.Module, class_id: int) -> torch.Tensor:
    """The classifier's weight vector w_k for class `class_id`, shape [F]."""
    head = get_classifier_head(model)
    if not (0 <= class_id < head.out_channels):
        raise ValueError(f"class_id {class_id} out of range [0, {head.out_channels})")
    return head.weight[class_id, :, 0, 0].detach().clone()


@contextmanager
def hook_pre_classifier_features(model: nn.Module):
    """Context manager yielding a dict that gains a 'features' entry (the
    spatial feature map [B, F, H, W] feeding `decoder.head`) after the next
    forward call inside the `with` block.
    """
    head = get_classifier_head(model)
    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(module, args):
        captured["features"] = args[0]

    handle = head.register_forward_pre_hook(_pre_hook)
    try:
        yield captured
    finally:
        handle.remove()


@torch.no_grad()
def extract_pixel_features(model: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Run the model and return the spatial feature map feeding its classifier.

    `*inputs` is forwarded to `model(...)` as-is, so this works for both the
    single-branch model (`model(x)`) and the context-branch model
    (`model(local_x, context_x)`).

    Returns
    -------
    torch.Tensor
        Feature map of shape [B, F, H, W], already at the input's spatial
        resolution.
    """
    model.eval()
    with hook_pre_classifier_features(model) as captured:
        model(*inputs)
        return captured["features"]


def extract_pooled_features(model: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Global-average-pooled features: phi(x) in the neural-PCA method. Shape [B, F]."""
    feats = extract_pixel_features(model, *inputs)
    return feats.mean(dim=(2, 3))
