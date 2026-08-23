"""Tests for training/segmentation.py's F1 (IG aux head) and F4 (decoder dropout)
levers, plus their round-trip through training/builders.py checkpoint save/load."""

import torch

from vision_backend.model.features import get_classifier_head, hook_pre_classifier_features
from vision_backend.training.builders import (
    build_simmim_segmentation_model,
    load_segmentation_model_from_checkpoint,
)
from vision_backend.training.segmentation import (
    ContextAwareSegmentationModel,
    LightweightSegmentationDecoder,
    SingleBranchSegmentationModel,
)

SIMMIM_CONFIG = {
    "model_kind": "simmim",
    "in_channels": 1,
    "global_base_grid": 4,
    "window_size": 8,
    "decoder_channels": 16,
    "num_classes": 3,
}


def _decoder_inputs(decoder_channels=16, size=32):
    return dict(
        bottleneck=torch.randn(1, decoder_channels, size // 8, size // 8),
        skip8=torch.randn(1, decoder_channels, size // 4, size // 4),
        skip4=torch.randn(1, decoder_channels, size // 2, size // 2),
        skip2=torch.randn(1, decoder_channels, size, size),
        output_size=(size * 2, size * 2),
    )


# --------------------------------------------------------------------------
# F1: IG aux head
# --------------------------------------------------------------------------
def test_decoder_default_returns_plain_dc_tensor():
    """Neutral default (num_classes_ig=None): forward() returns a bare Tensor,
    exactly today's contract -- every existing caller (mars-inference,
    uncertainty/, pc_align/) depends on this never changing."""
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3,
    )
    out = decoder(**_decoder_inputs())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 64, 64)


def test_decoder_return_ig_true_without_ig_head_returns_none_second():
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3,
    )
    dc_logits, ig_logits = decoder(**_decoder_inputs(), return_ig=True)
    assert dc_logits.shape == (1, 3, 64, 64)
    assert ig_logits is None


def test_decoder_with_ig_head_returns_both_logits_at_input_resolution():
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=14, num_classes_ig=5,
    )
    dc_logits, ig_logits = decoder(**_decoder_inputs(), return_ig=True)
    assert dc_logits.shape == (1, 14, 64, 64)
    assert ig_logits.shape == (1, 5, 64, 64)


def test_decoder_with_ig_head_default_call_still_returns_plain_dc_tensor():
    """num_classes_ig set at construction time (e.g. loaded from a checkpoint
    trained with the aux head) must NOT change plain `model(x)` behavior --
    only an explicit return_ig=True does."""
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=14, num_classes_ig=5,
    )
    out = decoder(**_decoder_inputs())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 14, 64, 64)


def test_ig_head_param_increase_is_small_and_bounded():
    decoder_channels = 16
    base = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=decoder_channels, num_classes=14,
    )
    with_ig = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=decoder_channels, num_classes=14, num_classes_ig=5,
    )
    base_params = sum(p.numel() for p in base.parameters())
    with_ig_params = sum(p.numel() for p in with_ig.parameters())
    expected_head_ig_params = decoder_channels * 5 + 5  # 1x1 conv weight + bias
    assert with_ig_params - base_params == expected_head_ig_params


def test_single_branch_model_return_ig_true_shapes(monkeypatch=None):
    config = dict(SIMMIM_CONFIG, num_classes=14)
    model = build_simmim_segmentation_model({**config, "num_classes_ig": 5})
    model.eval()
    with torch.no_grad():
        dc_logits, ig_logits = model(torch.randn(1, 1, 256, 256), return_ig=True)
    assert dc_logits.shape == (1, 14, 256, 256)
    assert ig_logits.shape == (1, 5, 256, 256)


def test_single_branch_model_default_call_unaffected_by_ig_head():
    config = dict(SIMMIM_CONFIG, num_classes=14)
    model = build_simmim_segmentation_model({**config, "num_classes_ig": 5})
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 256, 256))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 14, 256, 256)


def test_checkpoint_round_trip_with_ig_head_infers_num_classes_ig(tmp_path):
    model = build_simmim_segmentation_model({**SIMMIM_CONFIG, "num_classes": 14, "num_classes_ig": 5})
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"model": dict(SIMMIM_CONFIG, num_classes=14), "data": {"num_classes": 14}},
        },
        path,
    )
    loaded, model_kind, num_classes, config = load_segmentation_model_from_checkpoint(path, device="cpu")
    assert num_classes == 14
    assert loaded.decoder.head_ig is not None
    assert loaded.decoder.head_ig.out_channels == 5
    with torch.no_grad():
        dc_logits, ig_logits = loaded(torch.randn(1, 1, 256, 256), return_ig=True)
    assert dc_logits.shape == (1, 14, 256, 256)
    assert ig_logits.shape == (1, 5, 256, 256)


def test_checkpoint_round_trip_without_ig_head_has_none_head_ig(tmp_path):
    model = build_simmim_segmentation_model(dict(SIMMIM_CONFIG, num_classes=3))
    path = tmp_path / "ckpt.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": {"model": SIMMIM_CONFIG, "data": {"num_classes": 3}}},
        path,
    )
    loaded, _, _, _ = load_segmentation_model_from_checkpoint(path, device="cpu")
    assert loaded.decoder.head_ig is None


# --------------------------------------------------------------------------
# F4: decoder dropout
# --------------------------------------------------------------------------
def test_dropout_zero_reproduces_current_outputs_exactly():
    torch.manual_seed(0)
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3, dropout=0.0,
    )
    decoder.train()
    inputs = _decoder_inputs()
    out_a = decoder(**inputs)
    out_b = decoder(**inputs)
    assert torch.equal(out_a, out_b)  # dropout(p=0) never randomizes anything


def test_dropout_active_in_train_identity_in_eval():
    torch.manual_seed(0)
    decoder = LightweightSegmentationDecoder(
        bottleneck_channels=16, skip8_channels=16, skip4_channels=16, skip2_channels=16,
        decoder_channels=16, num_classes=3, dropout=0.5,
    )
    inputs = _decoder_inputs()

    decoder.train()
    outs = [decoder(**inputs) for _ in range(5)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), "dropout(p=0.5) should vary across calls in train mode"

    decoder.eval()
    with torch.no_grad():
        eval_outs = [decoder(**inputs) for _ in range(5)]
    for o in eval_outs[1:]:
        assert torch.equal(eval_outs[0], o), "dropout must be identity (deterministic) in eval mode"


def test_decoder_head_still_hookable_with_dropout_and_ig_head_active():
    """model/features.py's pre-forward hook on decoder.head must keep working
    -- and, since dropout is identity at eval, must see the SAME feature map
    it would without dropout/IG at all (features.py's own documented guarantee)."""
    model = build_simmim_segmentation_model(
        {**SIMMIM_CONFIG, "num_classes": 3, "num_classes_ig": 2, "decoder_dropout": 0.3}
    )
    model.eval()
    head = get_classifier_head(model)
    assert head is model.decoder.head
    assert isinstance(head, torch.nn.Conv2d)

    x = torch.randn(1, 1, 256, 256)
    with hook_pre_classifier_features(model) as captured:
        with torch.no_grad():
            model(x)
    assert "features" in captured
    assert captured["features"].shape[0] == 1


def test_context_aware_model_accepts_ig_and_dropout_params():
    from vision_backend.model.model import ContextAwareConvNeXtSwinEncoder

    encoder = ContextAwareConvNeXtSwinEncoder(
        in_channels=1, local_base_channels=8, context_base_channels=4, context_dim=16,
        use_stage32=False, swin_depths=(1, 1, 1), swin_num_heads=(2, 2, 2), window_size=4,
    )
    model = ContextAwareSegmentationModel(
        encoder=encoder, num_classes=3, bottleneck_channels=64, skip8_channels=32,
        skip4_channels=16, skip2_channels=8, decoder_channels=16,
        num_classes_ig=2, decoder_dropout=0.2,
    )
    model.eval()
    with torch.no_grad():
        dc_logits, ig_logits = model(
            torch.randn(1, 1, 128, 128), torch.randn(1, 1, 128, 128), return_ig=True
        )
    assert dc_logits.shape[1] == 3
    assert ig_logits.shape[1] == 2
