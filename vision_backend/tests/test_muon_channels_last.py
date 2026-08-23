"""Muon must survive channels_last conv gradients.

Regression test for the stage-3 sweep failure: trials that sampled
``muon_scope='all'`` together with ``--channels-last`` died with

    RuntimeError: view size is not compatible with input tensor's size and
    stride (at least one dimension spans across two contiguous subspaces).

Muon flattens >=3-D updates via ``update.view(len(update), -1)``; under
channels_last a conv weight's gradient is not contiguous in NCHW order, so the
view fails. It is also a correctness trap -- a successful view would have
flattened in NHWC order, feeding Newton-Schulz a permuted matrix.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vision_backend.model.optimizers import build_routed_muon_nadam_optimizer


class _ConvNet(nn.Module):
    """Has 4-D convs (Muon's problem case) and a 2-D linear (its normal case)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.dw = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(self.conv(x))
        return self.head(x.mean(dim=(2, 3)))


def _step(memory_format, scope):
    torch.manual_seed(0)
    model = _ConvNet().to(memory_format=memory_format)
    opt = build_routed_muon_nadam_optimizer(
        model, muon_lr=1e-3, nadam_lr=1e-4, muon_scope=scope
    )
    x = torch.randn(2, 3, 16, 16).to(memory_format=memory_format)
    loss = model(x).square().mean()
    loss.backward()
    opt.step()
    return model


@pytest.mark.parametrize("scope", ["matrix", "all"])
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last]
)
def test_step_survives_memory_format(memory_format, scope):
    """The combination that crashed the sweep must now complete a step."""
    model = _step(memory_format, scope)
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} became non-finite"


def test_channels_last_and_contiguous_agree_for_conv_muon():
    """Same update regardless of layout -- the flattening must not depend on it."""
    a = _step(torch.contiguous_format, "all")
    b = _step(torch.channels_last, "all")
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        torch.testing.assert_close(
            pa.to(memory_format=torch.contiguous_format),
            pb.to(memory_format=torch.contiguous_format),
            rtol=1e-4,
            atol=1e-6,
            msg=f"{na} diverged between memory formats",
        )
