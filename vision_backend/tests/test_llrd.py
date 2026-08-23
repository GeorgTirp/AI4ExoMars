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
    lrs = [g["lr"] for g in optimizer.param_groups]
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
