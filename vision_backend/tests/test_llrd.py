"""Tests for model/optimizers.py's F3 lever: layer-wise LR decay (LLRD)."""

import pytest
import torch

from vision_backend.model import optimizers as optim_mod
from vision_backend.model.optimizers import (
    _LLRD_MAX_DEPTH,
    build_routed_muon_nadam_optimizer,
    create_optimizer,
    llrd_depth_rank,
    llrd_lr_scale,
)
from vision_backend.training.builders import build_simmim_segmentation_model

SIMMIM_CONFIG = {
    "model_kind": "simmim",
    "in_channels": 1,
    "global_base_grid": 4,
    "window_size": 8,
    "decoder_channels": 16,
    "num_classes": 3,
}


def _model():
    torch.manual_seed(0)
    return build_simmim_segmentation_model(SIMMIM_CONFIG)


# --------------------------------------------------------------------------
# llrd_depth_rank / llrd_lr_scale
# --------------------------------------------------------------------------
def test_depth_rank_matches_hybrid_encoder_component_order():
    assert llrd_depth_rank("encoder.stem.weight") == 0
    assert llrd_depth_rank("encoder.s1.0.dwconv.weight") == 1
    assert llrd_depth_rank("encoder.down1.conv.weight") == 2
    assert llrd_depth_rank("encoder.s2.0.dwconv.weight") == 3
    assert llrd_depth_rank("encoder.down2.conv.weight") == 4
    assert llrd_depth_rank("encoder.s3.0.attn.qkv.weight") == 5
    assert llrd_depth_rank("encoder.down3.conv.weight") == 6
    assert llrd_depth_rank("encoder.s4.0.attn.qkv.weight") == 7
    assert llrd_depth_rank("encoder.norm.weight") == 7
    assert llrd_depth_rank("decoder.head.weight") == _LLRD_MAX_DEPTH
    assert llrd_depth_rank("decoder.head_ig.weight") == _LLRD_MAX_DEPTH
    assert llrd_depth_rank("decoder.bottleneck_proj.0.weight") == _LLRD_MAX_DEPTH


def test_unknown_encoder_param_falls_back_to_rank_zero():
    assert llrd_depth_rank("encoder.some_future_component.weight") == 0


def test_lr_scale_llrd_one_is_always_one():
    assert llrd_lr_scale("encoder.stem.weight", 1.0) == 1.0
    assert llrd_lr_scale("decoder.head.weight", 1.0) == 1.0


def test_lr_scale_monotonically_increases_with_depth():
    llrd = 0.8
    scale_stem = llrd_lr_scale("encoder.stem.weight", llrd)
    scale_s1 = llrd_lr_scale("encoder.s1.0.weight", llrd)
    scale_s4 = llrd_lr_scale("encoder.s4.0.weight", llrd)
    scale_head = llrd_lr_scale("decoder.head.weight", llrd)
    assert scale_stem < scale_s1 < scale_s4 < scale_head == 1.0


# --------------------------------------------------------------------------
# create_optimizer (plain AdamW/NAdam path)
# --------------------------------------------------------------------------
def test_create_optimizer_llrd_one_reproduces_uniform_lr():
    model = _model()
    base_lr = 1e-3
    optimizer = create_optimizer(model, lr=base_lr, weight_decay=0.01, use_muon=False, llrd=1.0)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(base_lr)


def test_create_optimizer_llrd_less_than_one_every_param_in_exactly_one_group():
    model = _model()
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.01, use_muon=False, llrd=0.8)
    grouped_params = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(grouped_params) == n_trainable
    assert len(set(id(p) for p in grouped_params)) == n_trainable  # no duplicates


def test_create_optimizer_llrd_less_than_one_lrs_monotonically_nondecreasing():
    model = _model()
    optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.01, use_muon=False, llrd=0.8)
    # Groups come as [decayed by depth] + [undecayed by depth] (norms/biases are
    # split out of weight decay), so LLRD's ordering holds within each bucket.
    for wd in (0.01, 0.0):
        lrs = [g["lr"] for g in optimizer.param_groups if g["weight_decay"] == wd]
        assert lrs == sorted(lrs)
        assert lrs[0] < lrs[-1]  # actually discounts something
        assert lrs[-1] == pytest.approx(1e-3)  # decoder/head group keeps full LR


# --------------------------------------------------------------------------
# build_routed_muon_nadam_optimizer
# --------------------------------------------------------------------------
def test_routed_optimizer_llrd_one_reproduces_uniform_lrs_per_suboptimizer():
    model = _model()
    optimizer = build_routed_muon_nadam_optimizer(
        model, muon_lr=0.01, nadam_lr=1e-4, llrd=1.0,
    )
    muon_opt, nadam_opt = optimizer.optimizers
    for group in muon_opt.param_groups:
        assert group["lr"] == pytest.approx(0.01)
    for group in nadam_opt.param_groups:
        assert group["lr"] == pytest.approx(1e-4)


def test_routed_optimizer_llrd_scales_within_each_suboptimizer_independently():
    model = _model()
    optimizer = build_routed_muon_nadam_optimizer(
        model, muon_lr=0.01, nadam_lr=1e-4, llrd=0.8,
    )
    muon_opt, nadam_opt = optimizer.optimizers

    muon_lrs = [g["lr"] for g in muon_opt.param_groups]
    assert all(lr <= 0.01 + 1e-12 for lr in muon_lrs)
    assert muon_lrs == sorted(muon_lrs)

    nadam_lrs = [g["lr"] for g in nadam_opt.param_groups]
    assert all(lr <= 1e-4 + 1e-15 for lr in nadam_lrs)


def test_routed_optimizer_llrd_every_param_lands_in_exactly_one_group():
    model = _model()
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    optimizer = build_routed_muon_nadam_optimizer(
        model, muon_lr=0.01, nadam_lr=1e-4, llrd=0.7,
    )
    all_params = [p for opt in optimizer.optimizers for g in opt.param_groups for p in g["params"]]
    assert len(all_params) == n_trainable
    assert len(set(id(p) for p in all_params)) == n_trainable


def test_routed_optimizer_fallback_path_uses_nadam_lr_not_muon_lr(monkeypatch):
    """Regression guard: the muon-unavailable fallback rebuilds its param
    groups from nadam_lr, not the already muon_lr-scaled `muon_groups` --
    reusing those would silently apply muon-scale LRs through NAdam."""
    monkeypatch.setattr(optim_mod, "_HAS_MUON", False)
    model = _model()
    optimizer = build_routed_muon_nadam_optimizer(
        model, muon_lr=10.0, nadam_lr=1e-4, llrd=1.0, require_muon=False,
    )
    muon_fallback_opt, nadam_opt = optimizer.optimizers
    for group in muon_fallback_opt.param_groups:
        assert group["lr"] == pytest.approx(1e-4)  # nadam_lr, NOT 10.0


# --------------------------------------------------------------------------
# Muon on transformer matrices only, Adam-matched LR, NAdamW for the rest
# --------------------------------------------------------------------------
def _simmim_model():
    from vision_backend.training.builders import build_simmim_segmentation_model

    return build_simmim_segmentation_model({
        "model_kind": "simmim", "in_channels": 1, "global_base_grid": 4,
        "window_size": 8, "decoder_channels": 16, "num_classes": 3,
    })


def _routed_transformer(model):
    return build_routed_muon_nadam_optimizer(
        model, muon_lr=1e-4, muon_weight_decay=5e-5, nadam_lr=1e-4, nadam_weight_decay=5e-5,
        muon_scope="transformer", muon_lr_mode="match_adam",
    )


def test_transformer_scope_routes_exactly_the_attention_and_mlp_matrices():
    model = _simmim_model()
    muon_opt, _ = _routed_transformer(model).optimizers
    muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    expected = {
        id(p) for n, p in model.named_parameters()
        if (".s3." in n or ".s4." in n) and p.ndim == 2 and "relative_position" not in n
    }
    assert muon_ids == expected and len(expected) == 32


def test_match_adam_uses_moonlight_scale_and_keeps_the_per_step_decay():
    import math

    from vision_backend.model.optimizers import muon_adam_matched_scale

    assert muon_adam_matched_scale(torch.Size([384, 384])) == pytest.approx(0.2 * math.sqrt(384))
    # SingleDeviceMuon already scales tall matrices by sqrt(A/B) = 2 here
    assert muon_adam_matched_scale(torch.Size([1536, 384])) == pytest.approx(0.2 * math.sqrt(1536) / 2)
    muon_opt, _ = _routed_transformer(_simmim_model()).optimizers
    for g in muon_opt.param_groups:
        c = muon_adam_matched_scale(g["params"][0].shape)
        assert g["lr"] == pytest.approx(1e-4 * c)
        assert g["lr"] * g["weight_decay"] == pytest.approx(1e-4 * 5e-5)


def test_routed_nadam_is_decoupled_and_never_decays_norms_or_biases():
    _, nadam_opt = _routed_transformer(_simmim_model()).optimizers
    assert nadam_opt.defaults["decoupled_weight_decay"] is True
    for g in nadam_opt.param_groups:
        if any(p.ndim <= 1 for p in g["params"]):
            assert g["weight_decay"] == 0.0
        else:
            assert g["weight_decay"] == pytest.approx(5e-5)
