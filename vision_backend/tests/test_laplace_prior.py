"""Diagonal Laplace prior: Fisher estimate, penalty value / gradient, scope,
and the guard that the model starts at the prior mean."""

import pytest
import torch

from vision_backend.laplace_prior import (
    DiagonalLaplacePrior,
    mc_fisher_diagonal_segmentation,
    quadratic_form,
    save_prior,
)
from vision_backend.training.builders import build_simmim_segmentation_model

CFG = {"model_kind": "simmim", "in_channels": 1, "global_base_grid": 8, "window_size": 8,
       "decoder_channels": 16, "num_classes": 3}


def _seg(seed=0):
    torch.manual_seed(seed)
    return build_simmim_segmentation_model(dict(CFG))


def _prior(tmp_path, seg, seed=1):
    g = torch.Generator().manual_seed(seed)
    fisher = {n: torch.rand(p.shape, generator=g) for n, p in seg.named_parameters()}
    save_prior(tmp_path / "prior.pt", dict(seg.named_parameters()), fisher, source_checkpoint="test")
    return fisher


def test_penalty_value_and_gradient(tmp_path):
    seg = _seg()
    fisher = _prior(tmp_path, seg)
    prior = DiagonalLaplacePrior(tmp_path / "prior.pt", seg, tau=3.0, delta=0.5)
    assert float(prior.penalty()) == 0.0
    mean = {n: p.detach().clone() for n, p in seg.named_parameters()}
    with torch.no_grad():
        for p in seg.parameters():
            p.add_(0.01 * torch.randn_like(p))
    pen = prior.penalty()
    expected = sum(0.5 * ((3.0 * fisher[n] + 0.5) * (p.detach() - mean[n]) ** 2).sum()
                   for n, p in seg.named_parameters())
    torch.testing.assert_close(pen, expected.float(), rtol=1e-4, atol=0)
    pen.backward()
    name, p = next(iter(seg.named_parameters()))
    torch.testing.assert_close(p.grad, (3.0 * fisher[name] + 0.5) * (p.detach() - mean[name]))
    theta = {n: q.detach() for n, q in seg.named_parameters()}
    expected_q = sum(float(0.5 * (fisher[n] * (theta[n] - mean[n]) ** 2).sum()) for n in theta)
    assert quadratic_form(theta, {"mean": mean, "fisher": fisher}) == pytest.approx(expected_q, rel=1e-5)


def test_encoder_scope_leaves_the_decoder_free(tmp_path):
    seg = _seg()
    _prior(tmp_path, seg)
    prior = DiagonalLaplacePrior(tmp_path / "prior.pt", seg, tau=1.0, scope="encoder")
    covered = {id(p) for p, _, _ in prior.terms}
    assert all(id(p) in covered for n, p in seg.named_parameters() if n.startswith("encoder."))
    assert not any(id(p) in covered for n, p in seg.named_parameters() if n.startswith("decoder."))
    with torch.no_grad():
        seg.decoder.head.weight.add_(1.0)
    assert float(prior.penalty()) == 0.0


def test_model_must_start_at_the_prior_mean(tmp_path):
    seg = _seg()
    _prior(tmp_path, seg)
    with pytest.raises(ValueError):
        DiagonalLaplacePrior(tmp_path / "prior.pt", _seg(seed=1), tau=1.0)


def test_mc_fisher_is_nonnegative_and_scales_with_pixels():
    seg = _seg()
    x = torch.randn(1, 1, 256, 256)
    full = {"image": x, "label": torch.zeros(1, 256, 256, dtype=torch.long)}
    half = {"image": x, "label": torch.zeros(1, 256, 256, dtype=torch.long)}
    half["label"][..., 128:] = 255  # ignored
    torch.manual_seed(0)
    f_full, n = mc_fisher_diagonal_segmentation(seg, [full] * 16, device=torch.device("cpu"), log_every=0)
    torch.manual_seed(0)
    f_half, _ = mc_fisher_diagonal_segmentation(seg, [half] * 16, device=torch.device("cpu"), log_every=0)
    assert n == 16
    assert set(f_full) == {n for n, p in seg.named_parameters()}
    assert all(bool((f >= 0).all()) for f in f_full.values())
    w = "decoder.head.weight"
    # the head's Fisher sums over the labelled pixels: half the pixels, about half the Fisher
    ratio = float(f_half[w].sum() / f_full[w].sum())
    assert 0.3 < ratio < 0.7
