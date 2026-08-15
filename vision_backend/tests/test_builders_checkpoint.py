"""Tests for training/builders.py's checkpoint -> model reconstruction."""

import pytest
import torch

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


def _save_checkpoint(path, model_config=SIMMIM_CONFIG, drop_num_classes_from_config=True):
    model = build_simmim_segmentation_model(model_config)
    saved_model_cfg = dict(model_config)
    if drop_num_classes_from_config:
        saved_model_cfg.pop("num_classes", None)
    checkpoint = {
        "stage": "stage3_segmentation_finetune",
        "epoch": 1,
        "model_state": model.state_dict(),
        "metrics": {"val_miou": 0.42},
        "config": {"model": saved_model_cfg, "data": {"num_classes": model_config["num_classes"]}},
    }
    torch.save(checkpoint, path)
    return model


def test_load_segmentation_model_from_checkpoint_simmim(tmp_path):
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path)

    model, model_kind, num_classes, config = load_segmentation_model_from_checkpoint(path, device="cpu")

    assert model_kind == "simmim"
    assert num_classes == 3
    assert config["model"]["model_kind"] == "simmim"
    assert not model.training  # eval() was called

    with torch.no_grad():
        out = model(torch.randn(1, 1, 256, 256))
    assert out.shape == (1, 3, 256, 256)


def test_num_classes_inferred_from_head_weight_when_absent_from_config(tmp_path):
    """The real stage3 script doesn't save num_classes in config['model'] -- head
    shape must be the source of truth."""
    path = tmp_path / "ckpt.pt"
    model = _save_checkpoint(path, drop_num_classes_from_config=True)

    _, _, num_classes, _ = load_segmentation_model_from_checkpoint(path, device="cpu")
    assert num_classes == model.decoder.head.out_channels == 3


def test_load_segmentation_model_from_checkpoint_unknown_kind_raises(tmp_path):
    path = tmp_path / "ckpt.pt"
    model = build_simmim_segmentation_model(SIMMIM_CONFIG)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"model": {"model_kind": "not_a_real_kind"}, "data": {"num_classes": 3}},
        },
        path,
    )
    with pytest.raises(ValueError, match="Unknown model_kind"):
        load_segmentation_model_from_checkpoint(path, device="cpu")
