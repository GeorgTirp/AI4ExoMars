"""End-to-end integration test for train_stage3_segmentation_finetune.py with
every Stage-3 performance lever (F1-F5) turned on simultaneously, on a tiny
synthetic dataset. Unit tests elsewhere cover each lever in isolation; this
one exists to catch wiring mistakes between CLI args -> build_config() ->
train_stage() that only show up when they're actually run together (new
kwarg name typos, config-key mismatches, shape mismatches between the IG
head/priors/dc_to_ig and a non-14 num_classes, EMA + llrd + IG interacting,
etc). Two epochs, 1 train batch, on CPU -- runs in a few seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import Affine

from vision_backend.model.hybrid_encoder import HybridEncoder
from vision_backend.train_stage3_segmentation_finetune import build_config, parse_args, train_stage


@pytest.fixture
def synthetic_stage3_setup(tmp_path):
    size = 512
    imagery = np.full((size, size), 150, dtype=np.uint8)
    labels = np.zeros((size, size), dtype=np.uint8)
    third = size // 3
    labels[:, :third] = 1
    labels[:, third:2 * third] = 2
    labels[:, 2 * third:] = 3

    imagery_path = tmp_path / "imagery.tif"
    label_path = tmp_path / "labels.tif"
    transform = Affine.identity()
    for path, data in ((imagery_path, imagery), (label_path, labels)):
        with rasterio.open(
            str(path), "w", driver="GTiff", height=size, width=size, count=1,
            dtype=np.uint8, crs="EPSG:4326", transform=transform, nodata=0,
        ) as dst:
            dst.write(data, 1)

    manifest_path = tmp_path / "crops.csv"
    manifest_path.write_text(
        "col,row,size,split\n"
        "0,0,256,train\n"
        "128,128,256,train\n"
        "256,128,256,val\n"
    )

    loader_config_path = tmp_path / "loader_config.json"
    loader_config_path.write_text(json.dumps({
        "manifest_path": str(manifest_path),
        "imagery_path": str(imagery_path),
        "label_path": str(label_path),
        "num_classes": 3,
    }))

    dc_to_ig_path = tmp_path / "dc_to_ig.json"
    dc_to_ig_path.write_text(json.dumps({"dc_to_ig": {"1": 1, "2": 1, "3": 2}}))

    encoder = HybridEncoder(in_channels=1, global_base_grid=4, window_size=8, drop_path=0.0)
    encoder_ckpt_path = tmp_path / "encoder.pt"
    torch.save(
        {"model_state": {f"encoder.{k}": v for k, v in encoder.state_dict().items()}},
        encoder_ckpt_path,
    )

    return {
        "loader_config_path": loader_config_path,
        "dc_to_ig_path": dc_to_ig_path,
        "encoder_ckpt_path": encoder_ckpt_path,
        "checkpoint_dir": tmp_path / "checkpoints",
        "history_path": tmp_path / "history.csv",
    }


def _parse_args_with(monkeypatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["train_stage3_segmentation_finetune.py"] + argv)
    return parse_args()


def test_all_levers_together_train_stage_runs_end_to_end(monkeypatch, synthetic_stage3_setup):
    s = synthetic_stage3_setup
    argv = [
        "--model-kind", "simmim",
        "--loader-factory", "vision_backend.seg_dataset:create_segmentation_dataloaders",
        "--loader-config-path", str(s["loader_config_path"]),
        "--batch-size", "1",
        "--num-workers", "0",
        "--epochs", "2",
        "--num-classes", "3",
        "--ignore-index", "255",
        "--encoder-checkpoint", str(s["encoder_ckpt_path"]),
        "--global-base-grid", "4",
        "--window-size", "8",
        "--decoder-channels", "8",
        "--decoder-dropout", "0.2",
        "--freeze-encoder-epochs", "0",
        "--use-muon",
        "--muon-lr", "0.01",
        "--nadam-lr", "0.0001",
        "--llrd", "0.8",
        "--ema-decay", "0.9",
        "--loss-kind", "balanced_softmax",
        "--class-weight-scheme", "none",
        "--ig-loss-weight", "0.5",
        "--num-classes-ig", "2",
        "--dc-to-ig-path", str(s["dc_to_ig_path"]),
        "--checkpoint-path", str(s["checkpoint_dir"] / "ckpt.pt"),
        "--history-path", str(s["history_path"]),
        "--seed", "0",
    ]
    args = _parse_args_with(monkeypatch, argv)
    config = build_config(args)

    final_metrics = train_stage(config)

    assert torch.isfinite(torch.tensor(final_metrics["best_val_miou"]))
    assert s["history_path"].exists()
    history_text = s["history_path"].read_text()
    assert "val_miou_ig" in history_text.splitlines()[0]  # F1: IG column present

    saved_checkpoints = list(s["checkpoint_dir"].glob("*.pt"))
    assert len(saved_checkpoints) == 1
    ckpt = torch.load(saved_checkpoints[0], map_location="cpu", weights_only=False)
    assert "ema_state" in ckpt  # F5: EMA state persisted for resume
    assert "decoder.head_ig.weight" in ckpt["model_state"]  # F1: IG head weights saved


def test_neutral_defaults_train_stage_matches_pre_lever_shape(monkeypatch, synthetic_stage3_setup):
    """Every new flag left at its default: history/checkpoint must look like
    today's plain run (no ig columns, no ema_state, no head_ig)."""
    s = synthetic_stage3_setup
    argv = [
        "--model-kind", "simmim",
        "--loader-factory", "vision_backend.seg_dataset:create_segmentation_dataloaders",
        "--loader-config-path", str(s["loader_config_path"]),
        "--batch-size", "1",
        "--num-workers", "0",
        "--epochs", "1",
        "--num-classes", "3",
        "--ignore-index", "255",
        "--encoder-checkpoint", str(s["encoder_ckpt_path"]),
        "--global-base-grid", "4",
        "--window-size", "8",
        "--decoder-channels", "8",
        "--freeze-encoder-epochs", "0",
        "--class-weight-scheme", "none",
        "--checkpoint-path", str(s["checkpoint_dir"] / "ckpt.pt"),
        "--history-path", str(s["history_path"]),
        "--seed", "0",
    ]
    args = _parse_args_with(monkeypatch, argv)
    config = build_config(args)

    train_stage(config)

    history_header = s["history_path"].read_text().splitlines()[0]
    assert "miou_ig" not in history_header

    saved_checkpoints = list(s["checkpoint_dir"].glob("*.pt"))
    ckpt = torch.load(saved_checkpoints[0], map_location="cpu", weights_only=False)
    assert "ema_state" not in ckpt
    assert "decoder.head_ig.weight" not in ckpt["model_state"]
