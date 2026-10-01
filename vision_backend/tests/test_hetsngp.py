"""Tests for model/hetsngp.py -- the per-pixel HetSNGP output layer
(Fortuin et al., TMLR 2022) -- and its wiring through the decoder, builders,
checkpoint loading and the model/features.py hooks."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_backend.model.features import (
    get_classifier_head,
    get_classifier_weight_vector,
    hook_pre_classifier_features,
)
from vision_backend.model.hetsngp import (
    HetSNGPHead2d,
    fit_laplace_covariance,
    orthogonal_random_features,
)
from vision_backend.training.builders import (
    build_simmim_segmentation_model,
    load_segmentation_model_from_checkpoint,
)

SIMMIM_CONFIG = {
    "model_kind": "simmim",
    "in_channels": 1,
    "global_base_grid": 4,
    "window_size": 8,
    "decoder_channels": 16,
    "num_classes": 3,
}
HEAD = {"head_type": "hetsngp", "num_inducing": 32, "num_factors": 2,
        "train_mc_samples": 4, "test_mc_samples": 8, "mc_chunk": 4}


def test_orthogonal_random_features_are_orthogonal_blocks_with_gaussian_row_norms():
    w = orthogonal_random_features(64, 16, torch.Generator().manual_seed(0))
    assert w.shape == (64, 16)
    gram = w[:16] @ w[:16].T
    assert (gram - torch.diag(torch.diag(gram))).abs().max() < 1e-4
    assert 3.0 < float(w.norm(dim=1).mean()) < 5.0  # chi_16 has mean ~3.9


def test_random_features_approximate_the_rbf_kernel():
    head = HetSNGPHead2d(8, 3, head_type="sngp", num_inducing=8192, normalize_input=False)
    h = 0.3 * torch.randn(2, 8, 1, 1, generator=torch.Generator().manual_seed(1))
    phi = head.random_features(h)[:, :, 0, 0]
    exact = torch.exp(-((h[0] - h[1]) ** 2).sum() / 2)
    assert abs(float((phi[0] * phi[1]).sum() - exact)) < 0.05
    assert abs(float((phi[0] ** 2).sum()) - 1.0) < 0.05


@pytest.mark.parametrize("head_type", ["het", "sngp", "hetsngp"])
def test_head_outputs_normalized_log_probabilities(head_type):
    head = HetSNGPHead2d(8, 5, head_type=head_type, num_inducing=64,
                         train_mc_samples=16, test_mc_samples=40, mc_chunk=16)
    x = torch.randn(2, 8, 4, 4)
    head.train()
    out = head(x)
    assert out.shape == (2, 5, 4, 4)
    if head_type != "sngp":  # sngp in training returns plain posterior-mode logits
        assert torch.allclose(out.logsumexp(dim=1), torch.zeros(2, 4, 4), atol=1e-5)
    head.eval()
    out = head(x)
    if head_type != "sngp":
        assert torch.allclose(out.logsumexp(dim=1), torch.zeros(2, 4, 4), atol=1e-5)


def test_vanishing_noise_recovers_the_plain_softmax():
    head = HetSNGPHead2d(8, 5, head_type="het", train_mc_samples=8)
    with torch.no_grad():
        head.scale_layer.weight.zero_()
        head.scale_layer.bias.zero_()
        head.diag_layer.weight.zero_()
        head.diag_layer.bias.fill_(-30.0)  # softplus -> 0, scale -> MIN_SCALE 1e-3
    x = torch.randn(2, 8, 3, 3)
    _, loc = head.phi_and_loc(x)
    assert torch.allclose(head.train()(x), F.log_softmax(loc, dim=1), atol=1e-2)


def test_larger_noise_flattens_the_predictive_distribution():
    head = HetSNGPHead2d(8, 5, head_type="het", test_mc_samples=512, mc_chunk=128).eval()
    x = torch.randn(1, 8, 3, 3)

    def entropy(bias):
        with torch.no_grad():
            head.diag_layer.bias.fill_(bias)
            lp = head(x)
        return float(-(lp.exp() * lp).sum(dim=1).mean())

    assert entropy(5.0) > entropy(-5.0) + 0.05


def test_mc_estimate_is_stable_across_seeds_at_large_s():
    head = HetSNGPHead2d(8, 4, head_type="het", test_mc_samples=4096, mc_chunk=1024).eval()
    x = torch.randn(1, 8, 2, 2)
    torch.manual_seed(0)
    a = head(x).exp()
    torch.manual_seed(1)
    b = head(x).exp()
    # two independent S-sample means differ with sd sqrt(2 p (1 - p) / S)
    sd = (2 * a * (1 - a) / 4096).sqrt().max()
    assert (a - b).abs().max() < 5 * sd


class _HeadOnly(nn.Module):
    """Model whose decoder.head sees the input directly, for the Laplace fit."""

    def __init__(self, head):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.head = head

    def forward(self, x):
        return self.decoder.head(x)


def test_laplace_variance_is_small_near_the_data_and_near_prior_far_from_it():
    torch.manual_seed(0)
    head = HetSNGPHead2d(4, 3, head_type="sngp", num_inducing=256)
    pattern = torch.tensor([1.0, -1.0, 0.5, -0.5]).view(1, 4, 1, 1)
    loader = [(pattern + 0.05 * torch.randn(4, 4, 8, 8), torch.zeros(4, 8, 8, dtype=torch.long))
              for _ in range(5)]
    stats = fit_laplace_covariance(_HeadOnly(head), head, loader, device="cpu", amp_dtype=None)
    assert head.gp_fitted and stats["labelled_pixels"] == 5 * 4 * 64
    near = head.gp_variance(head.random_features(pattern + 0.05 * torch.randn(1, 4, 2, 2))).mean()
    # after the per-pixel LayerNorm, -pattern is the antipode: squared distance 16, kernel ~3e-4
    far = head.gp_variance(head.random_features(-pattern.expand(1, 4, 2, 2))).mean()
    assert float(far) > 0.5          # prior marginal variance is E||Phi||^2 = 1
    assert float(near) < 0.05 * float(far)


def test_laplace_matches_the_closed_form_precision():
    torch.manual_seed(0)
    head = HetSNGPHead2d(3, 2, head_type="sngp", num_inducing=16, normalize_input=False)
    x = torch.randn(2, 3, 2, 2)
    fit_laplace_covariance(_HeadOnly(head), head, [(x, torch.zeros(2, 2, 2, dtype=torch.long))],
                           device="cpu", amp_dtype=None)
    phi, loc = head.phi_and_loc(x)
    p = torch.softmax(loc, dim=1)
    flat_phi = phi.permute(0, 2, 3, 1).reshape(-1, 16).double()
    for c in range(2):
        w = (p[:, c] * (1 - p[:, c])).reshape(-1, 1).double()
        precision = torch.eye(16, dtype=torch.float64) + (flat_phi * w).T @ flat_phi
        assert torch.allclose(head.covariance[c].double(), torch.linalg.inv(precision), atol=1e-5)


def test_fitted_covariance_round_trips_through_state_dict():
    a = HetSNGPHead2d(4, 3, head_type="hetsngp", num_inducing=8)
    a.covariance = torch.eye(8).repeat(3, 1, 1) * 0.5
    b = HetSNGPHead2d(4, 3, head_type="hetsngp", num_inducing=8)
    assert not b.gp_fitted
    b.load_state_dict(a.state_dict())
    assert b.gp_fitted and torch.equal(a.covariance, b.covariance)


def test_full_model_trains_checkpoints_and_keeps_feature_hooks(tmp_path):
    torch.manual_seed(0)
    cfg = dict(SIMMIM_CONFIG, uncertainty_head=HEAD)
    model = build_simmim_segmentation_model(cfg).train()
    x = torch.randn(2, 1, 256, 256)
    y = torch.randint(0, 3, (2, 256, 256))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    head = model.decoder.head
    for p in (head.weight, head.scale_layer.weight, head.diag_layer.weight,
              next(model.encoder.parameters())):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0

    torch.save({"model_state": model.state_dict(), "config": {"model": cfg}}, tmp_path / "c.pt")
    loaded, kind, num_classes, _ = load_segmentation_model_from_checkpoint(tmp_path / "c.pt")
    assert kind == "simmim" and num_classes == 3
    assert isinstance(get_classifier_head(loaded), HetSNGPHead2d)
    with hook_pre_classifier_features(loaded) as captured:
        out = loaded(x[:1])
    assert out.shape == (1, 3, 256, 256) and captured["features"].shape[1] == 16
    # upsampled output is renormalized: an exact per-pixel log predictive
    assert torch.allclose(out.logsumexp(dim=1), torch.zeros(1, 256, 256), atol=1e-5)
    with pytest.raises(ValueError, match="random-feature space"):
        get_classifier_weight_vector(loaded, 0)


def test_no_uncertainty_head_keeps_the_plain_conv_classifier():
    model = build_simmim_segmentation_model(dict(SIMMIM_CONFIG))
    assert type(model.decoder.head) is nn.Conv2d


@pytest.mark.parametrize("head_type", ["sngp", "hetsngp"])
def test_fitted_gp_variance_enters_the_eval_predictive(head_type):
    """Algorithm 2: once Sigma is fitted, eval samples beta too, so a far-from-data
    pixel gets a flatter predictive than the same head with Sigma switched off."""
    torch.manual_seed(0)
    head = HetSNGPHead2d(4, 3, head_type=head_type, num_inducing=256, test_mc_samples=2048, mc_chunk=512)
    pattern = torch.tensor([1.0, -1.0, 0.5, -0.5]).view(1, 4, 1, 1)
    loader = [(pattern + 0.05 * torch.randn(4, 4, 8, 8), torch.zeros(4, 8, 8, dtype=torch.long))]
    fit_laplace_covariance(_HeadOnly(head), head, loader, device="cpu", amp_dtype=None)
    with torch.no_grad():
        head.weight.mul_(20.0)  # confident posterior mode, so added variance is visible
        if head.use_het:  # silence the (random-init) data noise to isolate the GP term
            head.scale_layer.weight.zero_()
            head.scale_layer.bias.zero_()
            head.diag_layer.weight.zero_()
            head.diag_layer.bias.fill_(-30.0)
    far = -pattern.expand(1, 4, 2, 2)
    head.eval()
    with torch.no_grad():
        with_gp = head(far)
        fitted = head.covariance
        head.covariance = torch.zeros(0)
        without_gp = head(far)
        head.covariance = fitted
    assert torch.allclose(with_gp.logsumexp(dim=1), torch.zeros(1, 2, 2), atol=1e-5)

    def entropy(lp):
        return float(-(lp.exp() * lp).sum(dim=1).mean())

    assert entropy(with_gp) > entropy(without_gp) + 0.05
