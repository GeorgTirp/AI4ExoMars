"""§8.1, §8.2, §8.4 — encoder contract, param budget, resolution transfer."""

import pytest
import torch

from vision_backend.model.hybrid_encoder import HybridEncoder
from vision_backend.model.relpos import interpolate_relative_position_bias
from vision_backend.model.blocks_v2 import GlobalAttentionBlock

# §2.3 reference per-component counts
REFERENCE = {
    "stem": 1_632,
    "S1": 240_000,
    "S2": 922_000,
    "S3": 10_660_000,
    "S4": 14_180_000,
    "downsample": 1_550_000,
}


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    return HybridEncoder(in_channels=1).eval()


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_forward_contract_shapes(encoder, size):
    with torch.no_grad():
        out = encoder(torch.randn(1, 1, size, size))
    assert set(out) == {"s1", "s2", "s3", "s4", "pooled"}
    assert tuple(out["s1"].shape) == (1, 96, size // 4, size // 4)
    assert tuple(out["s2"].shape) == (1, 192, size // 8, size // 8)
    assert tuple(out["s3"].shape) == (1, 384, size // 16, size // 16)
    assert tuple(out["s4"].shape) == (1, 768, size // 32, size // 32)
    assert tuple(out["pooled"].shape) == (1, 768)
    assert torch.isfinite(out["s4"]).all()


def test_non_divisible_raises(encoder):
    with pytest.raises(ValueError):
        encoder(torch.randn(1, 1, 300, 256))


def test_param_budget(encoder):
    table = dict(encoder.param_table())
    total = table["encoder total"]
    assert 26_000_000 <= total <= 31_000_000, total
    for name, ref in REFERENCE.items():
        assert abs(table[name] - ref) <= 0.05 * ref + 256, (name, table[name], ref)


def test_relpos_interpolation_upsample_finite():
    # the 1024 -> 2048 path: bias table grid 32 -> 64
    heads = 24
    table = torch.randn((2 * 32 - 1) ** 2, heads)
    up = interpolate_relative_position_bias(table, 32, 64, heads)
    assert tuple(up.shape) == ((2 * 64 - 1) ** 2, heads)
    assert torch.isfinite(up).all()
    # identity when grids match
    same = interpolate_relative_position_bias(table, 32, 32, heads)
    assert torch.equal(same, table)


def test_resolution_transfer_end_to_end(encoder):
    # base grid is 32 (train 1024); forwarding at 512 uses grid 16 (interpolated)
    with torch.no_grad():
        out_train = encoder(torch.randn(1, 1, 1024, 1024))  # grid 32, no interp
        out_other = encoder(torch.randn(1, 1, 512, 512))     # grid 16, interp
    for out in (out_train, out_other):
        assert torch.isfinite(out["s4"]).all()
        assert torch.isfinite(out["pooled"]).all()
        assert float(out["s4"].std()) < 1e3


def test_global_block_requires_square():
    block = GlobalAttentionBlock(dim=768, num_heads=24, base_grid=8)
    with pytest.raises(ValueError):
        block(torch.randn(1, 768, 8, 16))
