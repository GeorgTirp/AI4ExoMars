"""Shared feature-extraction hooks for post-hoc analysis (uncertainty, neural PCA).

Both `SingleBranchSegmentationModel` and `ContextAwareSegmentationModel`
(`training/segmentation.py`) share one structural landmark regardless of encoder:
a final ``decoder.head`` -- a plain ``nn.Conv2d(decoder_channels, num_classes, 1)``.

Note the head runs at the *decoder's* resolution, not the input's:
`LightweightSegmentationDecoder.forward` classifies first and upsamples the
resulting logits, because a 1x1 conv and bilinear interpolation commute exactly
and doing the cheap operand first avoids materialising a
``decoder_channels x H x W`` activation at full input resolution on every
training step. `extract_pixel_features` therefore upsamples the hooked features
itself, so downstream analysis still receives an input-resolution map -- the
cost is paid only by the analysis paths that actually need it.

Hooking that conv's *input* gives, for any current or future decoder internals,
the same two things every downstream analysis in `uncertainty/` and `pc_align/`
needs:

- a per-pixel feature map already aligned with the input image (no extra
  upsampling bookkeeping), for spatial uncertainty maps;
- its global-average-pool, i.e. phi(x) in the neural-PCA method.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from .hetsngp import HetSNGPHead2d


def get_classifier_head(model: nn.Module) -> nn.Module:
    """Return the model's final per-pixel classifier (`model.decoder.head`).

    Either the plain 1x1 conv or a HetSNGP output layer; both take the same
    [B, F, H, W] decoder features as input, so the pre-classifier hook below
    captures the same representation for either.
    """
    decoder = getattr(model, "decoder", None)
    head = getattr(decoder, "head", None) if decoder is not None else None
    if not isinstance(head, (nn.Conv2d, HetSNGPHead2d)):
        raise ValueError(
            "Model has no `decoder.head` Conv2d classifier -- expected a "
            "SingleBranchSegmentationModel or ContextAwareSegmentationModel "
            "(training/segmentation.py)."
        )
    return head


def get_classifier_weight_vector(model: nn.Module, class_id: int) -> torch.Tensor:
    """The classifier's weight vector w_k for class `class_id`, shape [F]."""
    head = get_classifier_head(model)
    if isinstance(head, HetSNGPHead2d):
        raise ValueError(
            "HetSNGP head: class weights live in random-feature space, not in the "
            "decoder feature space, so there is no per-class feature-space vector w_k."
        )
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
        Feature map of shape [B, F, H, W] at the *first input's* spatial
        resolution. The decoder classifies before upsampling (see module
        docstring), so the hooked map is at the decoder's lower resolution and
        is resampled here with the same bilinear/align_corners=False settings
        the decoder uses for its logits -- keeping these features pixel-aligned
        with both the input image and the model's own output.
    """
    model.eval()
    with hook_pre_classifier_features(model) as captured:
        model(*inputs)
        features = captured["features"]
        output_size = inputs[0].shape[2:]
        if features.shape[2:] != output_size:
            features = nn.functional.interpolate(
                features, size=output_size, mode="bilinear", align_corners=False
            )
        return features


def extract_pooled_features(model: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Global-average-pooled features: phi(x) in the neural-PCA method. Shape [B, F]."""
    feats = extract_pixel_features(model, *inputs)
    return feats.mean(dim=(2, 3))
