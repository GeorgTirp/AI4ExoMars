"""§8.3 — SimMIM mask correctness (no leakage + masked-only loss)."""

import pytest
import torch
import torch.nn.functional as F

from vision_backend.model.hybrid_encoder import HybridEncoder
from vision_backend.model.simmim import SimMIMModel


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    encoder = HybridEncoder(in_channels=1, global_base_grid=8)  # 256² crops
    return SimMIMModel(encoder, in_channels=1, loss_type="l1").eval()


def _mask_px(model, mask):
    return (
        F.interpolate(mask.float().unsqueeze(1), scale_factor=model.mask_unit, mode="nearest")
        .bool()
    )


def test_no_leakage_from_masked_pixels(model):
    torch.manual_seed(1)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    mask = out["mask"]
    mask_px = _mask_px(model, mask)

    # perturbing MASKED input pixels must not change the reconstruction
    x_masked_perturb = x.clone()
    x_masked_perturb[mask_px.expand_as(x)] += 5.0
    out_m = model(x_masked_perturb, mask=mask)
    assert torch.allclose(out_m["reconstruction"], out["reconstruction"], atol=1e-5)

    # perturbing UNMASKED input pixels must change the reconstruction
    x_unmasked_perturb = x.clone()
    x_unmasked_perturb[~mask_px.expand_as(x)] += 5.0
    out_u = model(x_unmasked_perturb, mask=mask)
    assert not torch.allclose(out_u["reconstruction"], out["reconstruction"], atol=1e-4)


def test_loss_is_masked_only_l1(model):
    torch.manual_seed(2)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    mask_px = _mask_px(model, out["mask"]).float()
    recon = out["reconstruction"]
    manual = (recon - x).abs().mul(mask_px).sum() / (mask_px.sum() * recon.shape[1])
    assert torch.allclose(out["loss"], manual, atol=1e-5)


def test_mask_ratio_in_range(model):
    torch.manual_seed(3)
    out = model(torch.randn(4, 1, 256, 256))
    ratio = float(out["mask"].float().mean())
    assert 0.5 < ratio < 0.7
