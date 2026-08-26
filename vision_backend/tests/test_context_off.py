"""`use_context=False` must DROP the context branch, not merely skip it.

If the sub-encoder and FiLM layers were still constructed and simply bypassed,
the no-context variants would carry dead parameters: the optimizer would hold
state for them, checkpoints would store them, and the V0-vs-V1 parameter delta
would no longer mean "what the context mechanism costs". So these tests assert
absence of parameters, not just that the forward runs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vision_backend.model.features import get_classifier_head
from vision_backend.training.builders import (
    build_context_encoder,
    build_context_segmentation_model,
)

SIZE = 256  # encoder requires H, W divisible by 256
NUM_CLASSES = 14


def _config(use_context: bool, **overrides):
    config = dict(
        in_channels=1, local_base_channels=52, context_base_channels=26,
        context_dim=256, use_stage32=True, swin_depths=(2, 2, 2),
        swin_num_heads=(4, 8, 16), window_size=8, drop_path=0.0,
        num_classes=NUM_CLASSES, decoder_channels=256, use_context=use_context,
    )
    config.update(overrides)
    return config


def _context_param_count(module) -> int:
    """Parameters belonging to the context mechanism (sub-encoder + FiLM)."""
    return sum(
        p.numel()
        for name, p in module.named_parameters()
        if "context_encoder" in name or ".film" in name or name.startswith("film")
    )


def test_context_off_builds_no_context_submodules():
    encoder = build_context_encoder(_config(False))
    assert encoder.context_encoder is None
    film = [name for name, _ in encoder.named_modules() if "film" in name]
    assert film == [], f"FiLM modules still present with context off: {film}"


def test_context_off_has_zero_context_parameters():
    """The assertion that matters: no dead weights, not just an unused path."""
    off = build_context_segmentation_model(_config(False))
    on = build_context_segmentation_model(_config(True))

    assert _context_param_count(off) == 0
    assert _context_param_count(on) > 0

    # and the whole-model delta is exactly the context machinery
    delta = sum(p.numel() for p in on.parameters()) - sum(
        p.numel() for p in off.parameters()
    )
    assert delta == _context_param_count(on), (
        "the on/off parameter difference is not accounted for by the context "
        "branch alone -- something else changed between the two builds"
    )


def test_context_off_runs_on_local_input_alone():
    model = build_context_segmentation_model(_config(False)).eval()
    x = torch.randn(2, 1, SIZE, SIZE)
    with torch.no_grad():
        out = model(x)                 # no context argument at all
        out_explicit_none = model(x, None)
    assert out.shape == (2, NUM_CLASSES, SIZE, SIZE)
    assert torch.equal(out, out_explicit_none)


def test_context_on_still_requires_a_context_crop():
    """The off switch must not weaken the on path's contract."""
    model = build_context_segmentation_model(_config(True)).eval()
    x = torch.randn(2, 1, SIZE, SIZE)
    with pytest.raises(ValueError, match="use_context=True"):
        model(x, None)


def test_context_off_trains_end_to_end():
    model = build_context_segmentation_model(_config(False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 1, SIZE, SIZE)
    target = torch.randint(0, NUM_CLASSES, (2, SIZE, SIZE))
    target[:, :8, :] = 255

    loss = nn.CrossEntropyLoss(ignore_index=255)(model(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    ), "some parameters received no gradient with context off"


def test_decoder_head_contract_holds_with_context_off():
    """features.py / uncertainty / pc_align / mars-inference all depend on this."""
    for use_context in (False, True):
        model = build_context_segmentation_model(_config(use_context)).eval()
        head = get_classifier_head(model)
        assert isinstance(head, nn.Conv2d)
        assert head.kernel_size == (1, 1)
        assert head.out_channels == NUM_CLASSES


def test_context_off_encoder_returns_the_full_pyramid():
    """The decoder indexes the encoder's outputs positionally; context off must
    not change how many feature maps come back."""
    off = build_context_encoder(_config(False)).eval()
    on = build_context_encoder(_config(True)).eval()
    x = torch.randn(1, 1, SIZE, SIZE)
    c = torch.randn(1, 1, SIZE, SIZE)
    with torch.no_grad():
        feats_off = off(x)
        feats_on = on(x, c)
    assert len(feats_off) == len(feats_on)
    assert [f.shape for f in feats_off] == [f.shape for f in feats_on]
