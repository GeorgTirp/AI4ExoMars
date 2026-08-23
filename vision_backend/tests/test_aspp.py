"""Tests for training/segmentation.py's M1 lever: ASPP at the encoder
bottleneck."""

import torch

from vision_backend.model.features import get_classifier_head, hook_pre_classifier_features
from vision_backend.training.builders import (
    build_simmim_segmentation_model,
    load_segmentation_model_from_checkpoint,
)
from vision_backend.training.segmentation import ASPP, LightweightSegmentationDecoder

SIMMIM_CONFIG = {
    "model_kind": "simmim",
    "in_channels": 1,
    "global_base_grid": 4,
    "window_size": 8,
    "decoder_channels": 8,
    "num_classes": 3,
}


def _decoder_inputs(decoder_channels=16, bottleneck_channels=16, size=32):
    return dict(
        bottleneck=torch.randn(1, bottleneck_channels, size // 8, size // 8),
        skip8=torch.randn(1, decoder_channels, size // 4, size // 4),
        skip4=torch.randn(1, decoder_channels, size // 2, size // 2),
        skip2=torch.randn(1, decoder_channels, size, size),
        output_size=(size * 2, size * 2),
    )


# --------------------------------------------------------------------------
# ASPP module itself
# --------------------------------------------------------------------------
def test_aspp_preserves_spatial_shape_and_projects_channels():
    aspp = ASPP(in_channels=32, out_channels=16, rates=(6, 12, 18))
    x = torch.randn(2, 32, 10, 10)
    out = aspp(x)
    assert out.shape == (2, 16, 10, 10)


def test_aspp_works_on_1x1_spatial_input():
    """The bottleneck can be as small as 1x1 for a tiny/heavily-downsampled
    crop; the image-pool branch's AdaptiveAvgPool2d(1) + upsample must not
    choke on a degenerate all-pixels-already-pooled case."""
    aspp = ASPP(in_channels=8, out_channels=4, rates=(2, 4))
    x = torch.randn(1, 8, 1, 1)
    out = aspp(x)
    assert out.shape == (1, 4, 1, 1)


def test_aspp_rates_control_number_of_dilated_branches():
    aspp = ASPP(in_channels=8, out_channels=4, rates=(1, 2, 3, 4, 5))
    assert len(aspp.dilated_branches) == 5


def test_aspp_gradients_are_finite():
    aspp = ASPP(in_channels=8, out_channels=4, rates=(6, 12))
    x = torch.randn(1, 8, 6, 6, requires_grad=True)
    out = aspp(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in aspp.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------
# Decoder wiring: off by default, on when requested, decoder.head unaffected
# --------------------------------------------------------------------------
def test_decoder_use_aspp_false_has_no_aspp_module():
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3, use_aspp=False,
    )
    assert decoder.aspp is None


def test_decoder_use_aspp_true_produces_correct_output_shape():
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3, use_aspp=True, aspp_rates=(2, 4),
    )
    out = decoder(**_decoder_inputs(decoder_channels=16, bottleneck_channels=16))
    assert out.shape == (1, 3, 64, 64)


def test_decoder_use_aspp_true_with_ig_head_produces_both_shapes():
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=14, num_classes_ig=5, use_aspp=True,
    )
    dc_logits, ig_logits = decoder(
        **_decoder_inputs(decoder_channels=16, bottleneck_channels=16), return_ig=True
    )
    assert dc_logits.shape == (1, 14, 64, 64)
    assert ig_logits.shape == (1, 5, 64, 64)


def test_single_branch_model_use_aspp_true_runs_end_to_end():
    config = dict(SIMMIM_CONFIG, use_aspp=True, aspp_rates=(2, 4))
    model = build_simmim_segmentation_model(config)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 256, 256))
    assert out.shape == (1, 3, 256, 256)


def test_decoder_head_still_hookable_with_aspp_active():
    model = build_simmim_segmentation_model(dict(SIMMIM_CONFIG, use_aspp=True))
    model.eval()
    head = get_classifier_head(model)
    assert head is model.decoder.head
    x = torch.randn(1, 1, 256, 256)
    with hook_pre_classifier_features(model) as captured:
        with torch.no_grad():
            model(x)
    assert "features" in captured


# --------------------------------------------------------------------------
# Neutral default: use_aspp=False is byte-identical to before ASPP existed
# --------------------------------------------------------------------------
def test_use_aspp_false_matches_no_aspp_construction_exactly():
    torch.manual_seed(0)
    with_flag_off = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3, use_aspp=False,
    )
    torch.manual_seed(0)
    without_flag_at_all = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3,
    )
    inputs = _decoder_inputs(decoder_channels=16, bottleneck_channels=16)
    out_a = with_flag_off(**inputs)
    out_b = without_flag_at_all(**inputs)
    assert torch.equal(out_a, out_b)


# --------------------------------------------------------------------------
# Checkpoint round-trip
# --------------------------------------------------------------------------
def test_checkpoint_round_trip_with_aspp_preserves_rates(tmp_path):
    config = dict(SIMMIM_CONFIG, num_classes=3, use_aspp=True, aspp_rates=(2, 4, 6, 8))
    model = build_simmim_segmentation_model(config)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": {"model": config, "data": {"num_classes": 3}}},
        path,
    )
    loaded, _, num_classes, _ = load_segmentation_model_from_checkpoint(path, device="cpu")
    assert num_classes == 3
    assert loaded.decoder.aspp is not None
    assert len(loaded.decoder.aspp.dilated_branches) == 4  # matches aspp_rates=(2,4,6,8)
    with torch.no_grad():
        out = loaded(torch.randn(1, 1, 256, 256))
    assert out.shape == (1, 3, 256, 256)


def test_checkpoint_round_trip_without_aspp_has_none_aspp(tmp_path):
    model = build_simmim_segmentation_model(dict(SIMMIM_CONFIG, num_classes=3))
    path = tmp_path / "ckpt.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": {"model": SIMMIM_CONFIG, "data": {"num_classes": 3}}},
        path,
    )
    loaded, _, _, _ = load_segmentation_model_from_checkpoint(path, device="cpu")
    assert loaded.decoder.aspp is None
