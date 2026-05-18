from __future__ import annotations

from pathlib import Path
from typing import Any

from model.model import (
    ContextAwareConvNeXtSwinEncoder,
    ContextAwareMaskedReconstructionPretrainer,
)

from .segmentation import ContextAwareSegmentationModel
from .utils import extract_state_dict, load_checkpoint, load_prefixed_state_dict


def build_context_encoder(model_config: dict[str, Any]) -> ContextAwareConvNeXtSwinEncoder:
    return ContextAwareConvNeXtSwinEncoder(
        in_channels=model_config.get("in_channels", 1),
        local_base_channels=model_config["local_base_channels"],
        context_base_channels=model_config["context_base_channels"],
        context_dim=model_config["context_dim"],
        use_stage32=model_config["use_stage32"],
        swin_depths=tuple(model_config["swin_depths"]),
        swin_num_heads=tuple(model_config["swin_num_heads"]),
        window_size=model_config["window_size"],
        drop_path=model_config.get("drop_path", 0.0),
    )


def build_context_pretrainer(model_config: dict[str, Any]) -> ContextAwareMaskedReconstructionPretrainer:
    encoder = build_context_encoder(model_config)
    bottleneck_channels = model_config["local_base_channels"] * (
        16 if model_config["use_stage32"] else 8
    )
    skip8_channels = model_config["local_base_channels"] * 4

    return ContextAwareMaskedReconstructionPretrainer(
        encoder=encoder,
        in_channels=model_config.get("in_channels", 1),
        bottleneck_channels=bottleneck_channels,
        decoder_channels=model_config["decoder_channels"],
        patch_size=model_config["mask_patch_size"],
        mask_ratio=model_config["mask_ratio"],
        bottleneck_index=-1,
        skip8_index=3,
        skip8_channels=skip8_channels,
        use_skip8=True,
        loss_type=model_config["loss_type"],
    )


def build_context_segmentation_model(model_config: dict[str, Any]) -> ContextAwareSegmentationModel:
    encoder = build_context_encoder(model_config)
    bottleneck_channels = model_config["local_base_channels"] * (
        16 if model_config["use_stage32"] else 8
    )
    skip8_channels = model_config["local_base_channels"] * 4
    skip4_channels = model_config["local_base_channels"] * 2
    skip2_channels = model_config["local_base_channels"]

    return ContextAwareSegmentationModel(
        encoder=encoder,
        num_classes=model_config["num_classes"],
        bottleneck_channels=bottleneck_channels,
        skip8_channels=skip8_channels,
        skip4_channels=skip4_channels,
        skip2_channels=skip2_channels,
        decoder_channels=model_config["decoder_channels"],
        bottleneck_index=-1,
        skip8_index=3,
        skip4_index=1,
        skip2_index=0,
    )


def load_encoder_from_pretrainer_checkpoint(
    torch_module,
    encoder_model,
    checkpoint_path: Path,
    *,
    strict: bool = True,
) -> None:
    checkpoint = load_checkpoint(torch_module, checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(
        checkpoint,
        "model_state",
        "student_state",
        "teacher_state",
    )
    try:
        load_prefixed_state_dict(
            encoder_model,
            state_dict,
            prefix="encoder.",
            strict=strict,
        )
        return
    except KeyError:
        pass

    try:
        load_prefixed_state_dict(
            encoder_model,
            state_dict,
            prefix="student.encoder.",
            strict=strict,
        )
        return
    except KeyError:
        pass

    encoder_model.load_state_dict(state_dict, strict=strict)
