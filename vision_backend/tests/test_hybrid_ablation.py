"""HybridEncoder architecture-ablation switches (s3_depth, s4_global): the
default must stay the frozen layout, the ablations must build, run and
round-trip through checkpoints."""

import torch

from vision_backend.model.blocks_v2 import GlobalAttentionBlock
from vision_backend.model.hybrid_encoder import HybridEncoder
from vision_backend.model.model import SwinTransformerBlock
from vision_backend.training.builders import (
    build_simmim_segmentation_model,
    load_segmentation_model_from_checkpoint,
)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_default_is_the_frozen_layout():
    enc = HybridEncoder()
    assert _n(enc) == 27_655_608  # the documented Model v2 encoder size
    assert len(enc.s3) == 6 and isinstance(enc.s4[1], GlobalAttentionBlock)


def test_ablations_build_and_run_at_512():
    x = torch.randn(1, 1, 512, 512)
    for kw in ({"s4_global": False}, {"s3_depth": 2}):
        enc = HybridEncoder(**kw).eval()
        with torch.no_grad():
            out = enc(x)
        assert out["s4"].shape == (1, 768, 16, 16) and out["s3"].shape == (1, 384, 32, 32)
    assert isinstance(HybridEncoder(s4_global=False).s4[1], SwinTransformerBlock)
    assert HybridEncoder(s4_global=False).s4[1].shift_size == 4
    assert len(HybridEncoder(s3_depth=2).s3) == 2


def test_ablated_checkpoint_round_trips(tmp_path):
    cfg = {"model_kind": "simmim", "in_channels": 1, "global_base_grid": 4, "window_size": 8,
           "decoder_channels": 16, "num_classes": 3, "hybrid_s3_depth": 2, "hybrid_s4_global": False}
    model = build_simmim_segmentation_model(cfg)
    torch.save({"model_state": model.state_dict(), "config": {"model": cfg}}, tmp_path / "c.pt")
    loaded, *_ = load_segmentation_model_from_checkpoint(tmp_path / "c.pt")
    assert len(loaded.encoder.s3) == 2 and isinstance(loaded.encoder.s4[1], SwinTransformerBlock)
