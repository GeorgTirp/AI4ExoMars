"""Spatial context fusion (context_fusion="xattn") vs the pooled FiLM path.

The FiLM path averages the whole 494 m context window into one vector, so it
cannot represent WHERE a pattern sits relative to the crop. The xattn path keeps
the context grid and lets the local bottleneck attend to it with positions in a
shared frame. These tests pin the properties that make that a fair, safe
replacement:

* the default ("film") builds exactly what it always did -- same modules, same
  initialisation -- so existing checkpoints and running comparisons are safe;
* "xattn" carries no dead weight: no pooled head, no FiLM layers;
* at initialisation "xattn" is the identity (the context has NO effect), so it
  starts from the no-context model rather than from noise;
* once trained it does use the context, and it is position-aware;
* its geometry survives a checkpoint round trip.
"""

from __future__ import annotations

import pytest
import torch

from vision_backend.model.model import _token_centres
from vision_backend.training.builders import (
    build_context_encoder,
    build_context_segmentation_model,
)

SIZE = 256  # the encoder requires H, W divisible by 256
BATCH = 2


def _config(**overrides):
    config = dict(
        in_channels=1, local_base_channels=8, context_base_channels=4,
        context_dim=64, use_stage32=True, swin_depths=(2, 2, 2),
        swin_num_heads=(4, 8, 16), window_size=8, drop_path=0.0,
        num_classes=14, decoder_channels=32, use_context=True,
    )
    config.update(overrides)
    return config


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    local = torch.randn(BATCH, 1, SIZE, SIZE, generator=g)
    context = torch.randn(BATCH, 1, SIZE, SIZE, generator=g)
    return local, context


def _open_context_path(model) -> None:
    """Stand-in for training: make the zero-initialised output projection
    non-zero, which is all it takes for the context to start mattering."""
    with torch.no_grad():
        model.encoder.context_xattn.out_proj.weight.normal_(0.0, 0.05)


def test_film_is_the_default_and_builds_what_it_always_did():
    encoder = build_context_encoder(_config())
    assert encoder.context_fusion == "film"
    assert encoder.context_encoder.proj is not None, "pooled head missing on the FiLM path"
    assert hasattr(encoder, "film_x2") and hasattr(encoder, "film_x6")
    assert not hasattr(encoder, "context_xattn")


def test_explicit_film_initialises_identically_to_the_default():
    """The option existing must not perturb FiLM's RNG consumption."""
    torch.manual_seed(0)
    default = build_context_segmentation_model(_config())
    torch.manual_seed(0)
    explicit = build_context_segmentation_model(_config(context_fusion="film"))
    a, b = default.state_dict(), explicit.state_dict()
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_xattn_carries_no_dead_weight():
    encoder = build_context_encoder(_config(context_fusion="xattn"))
    film = [name for name, _ in encoder.named_modules() if "film" in name]
    assert film == [], f"FiLM layers built on the xattn path: {film}"
    assert encoder.context_encoder.proj is None, "unused pooled head built on the xattn path"
    assert hasattr(encoder, "context_xattn")


def test_xattn_is_the_identity_at_initialisation():
    """The context must have exactly zero effect before any training."""
    torch.manual_seed(0)
    model = build_context_segmentation_model(_config(context_fusion="xattn")).eval()
    local, ctx_a = _inputs(0)
    _, ctx_b = _inputs(1)
    with torch.no_grad():
        out_a = model(local, ctx_a)
        out_b = model(local, ctx_b)
    assert torch.equal(out_a, out_b)


def test_xattn_uses_the_context_once_the_path_opens():
    torch.manual_seed(0)
    model = build_context_segmentation_model(_config(context_fusion="xattn")).eval()
    _open_context_path(model)
    local, ctx_a = _inputs(0)
    _, ctx_b = _inputs(1)
    with torch.no_grad():
        diff = (model(local, ctx_a) - model(local, ctx_b)).abs().max()
    assert diff > 1e-4, f"context has no effect after opening the path (max diff {diff})"


def test_xattn_is_position_aware():
    """Mirroring the context must change the answer.

    This is the property the pooled path lacks by construction: a global
    average cannot tell a ripple field on the west side of the window from the
    same field on the east side.
    """
    torch.manual_seed(0)
    model = build_context_segmentation_model(_config(context_fusion="xattn")).eval()
    _open_context_path(model)
    local, ctx = _inputs(0)
    with torch.no_grad():
        diff = (model(local, ctx) - model(local, torch.flip(ctx, dims=[-1]))).abs().max()
    assert diff > 1e-4, f"output unchanged when the context is mirrored (max diff {diff})"


def test_gradient_reaches_the_context_encoder_after_the_first_step():
    """At step 0 only the zero-initialised projection learns; one step later the
    context encoder itself receives gradient -- the path genuinely trains."""
    torch.manual_seed(0)
    model = build_context_segmentation_model(_config(context_fusion="xattn")).train()
    local, ctx = _inputs(0)
    target = torch.randint(0, 14, (BATCH, SIZE, SIZE))
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def context_grad_norm() -> float:
        return sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if "context_encoder" in n and p.grad is not None
        )

    opt.zero_grad()
    loss_fn(model(local, ctx), target).backward()
    assert context_grad_norm() == 0.0, "context encoder trained through a zero projection"
    assert float(model.encoder.context_xattn.out_proj.weight.grad.abs().sum()) > 0.0
    opt.step()

    opt.zero_grad()
    loss_fn(model(local, ctx), target).backward()
    assert context_grad_norm() > 0.0, "context encoder still receives no gradient"


def test_shared_frame_geometry():
    """Local tokens cover one crop width; context tokens cover extent_ratio crop
    widths; both are centred on the crop."""
    ly, lx = _token_centres(16, 16, 1.0, "cpu")
    cy, cx = _token_centres(16, 16, 4.0, "cpu")
    for grid in (ly, lx, cy, cx):
        assert abs(float(grid.mean())) < 1e-6, "grid not centred on the crop"
    assert float(lx.abs().max()) < 0.5
    assert 1.5 < float(cx.abs().max()) < 2.0
    # Row-major, matching tensor.flatten(2): x varies fastest.
    assert float(lx[1] - lx[0]) > 0 and float(ly[1] - ly[0]) == 0


def test_geometry_survives_a_checkpoint_round_trip():
    """The extent ratio is derived from the data at training time and must be
    rebuilt from the saved config, not silently reset to the default 4."""
    torch.manual_seed(0)
    config = _config(context_fusion="xattn", context_extent_ratio=3.0)
    model = build_context_segmentation_model(config).eval()
    _open_context_path(model)

    rebuilt = build_context_segmentation_model(dict(config)).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.encoder.context_xattn.extent_ratio == 3.0

    local, ctx = _inputs(0)
    with torch.no_grad():
        assert torch.equal(model(local, ctx), rebuilt(local, ctx))


def test_configs_written_before_the_option_existed_rebuild_as_film():
    legacy = _config()  # no context_fusion / context_extent_ratio keys
    assert "context_fusion" not in legacy
    assert build_context_encoder(legacy).context_fusion == "film"


def test_unknown_fusion_is_rejected():
    with pytest.raises(ValueError, match="context_fusion"):
        build_context_encoder(_config(context_fusion="concat"))
