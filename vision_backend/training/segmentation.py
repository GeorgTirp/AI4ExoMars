from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from vision_backend.model.model import ConvNeXtBlock, LayerNorm2d
    from vision_backend.model.hetsngp import HetSNGPHead2d
except ModuleNotFoundError:
    from model.model import ConvNeXtBlock, LayerNorm2d
    from model.hetsngp import HetSNGPHead2d


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (Chen et al., DeepLab v3) at the
    encoder bottleneck (M1: NOAH-H's own stated fix for a HiRISE framelet
    crop's limited field of view, applied at the stride-32 map so it's cheap).

    Parallel branches -- a 1x1 conv, one dilated 3x3 conv per rate in `rates`,
    and a global-average-pool + 1x1 conv branch upsampled back to the input's
    spatial size -- are concatenated and projected to `out_channels`. Dilation
    widens the receptive field without adding stride or parameters-per-pixel
    the way a deeper/strided stack would.
    """

    def __init__(self, in_channels: int, out_channels: int, *, rates: Sequence[int] = (6, 12, 18)):
        super().__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.dilated_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    LayerNorm2d(out_channels),
                    nn.GELU(),
                )
                for rate in rates
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        num_branches = 2 + len(rates)  # 1x1 + dilated branches + image pool
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        branches = [self.branch_1x1(x), *[branch(x) for branch in self.dilated_branches], pooled]
        return self.project(torch.cat(branches, dim=1))


class LightweightSegmentationDecoder(nn.Module):
    def __init__(
        self,
        *,
        bottleneck_channels: int,
        skip8_channels: int,
        skip4_channels: int,
        skip2_channels: int,
        decoder_channels: int,
        num_classes: int,
        num_classes_ig: Optional[int] = None,
        dropout: float = 0.0,
        use_aspp: bool = False,
        aspp_rates: Sequence[int] = (6, 12, 18),
        uncertainty_head: Optional[dict] = None,
    ):
        super().__init__()

        # M1: optional, applied to the bottleneck feature map before it enters
        # bottleneck_proj below -- bottleneck_proj and everything downstream
        # (including decoder.head) are completely unchanged either way.
        self.aspp = ASPP(bottleneck_channels, bottleneck_channels, rates=aspp_rates) if use_aspp else None

        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(bottleneck_channels, decoder_channels, kernel_size=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            ConvNeXtBlock(decoder_channels),
        )
        self.skip8_proj = nn.Sequential(
            nn.Conv2d(skip8_channels, decoder_channels, kernel_size=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.skip4_proj = nn.Sequential(
            nn.Conv2d(skip4_channels, decoder_channels, kernel_size=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.skip2_proj = nn.Sequential(
            nn.Conv2d(skip2_channels, decoder_channels, kernel_size=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
        )

        self.fuse8 = nn.Sequential(
            ConvNeXtBlock(decoder_channels),
            ConvNeXtBlock(decoder_channels),
        )
        self.fuse4 = nn.Sequential(
            ConvNeXtBlock(decoder_channels),
            ConvNeXtBlock(decoder_channels),
        )
        self.fuse2 = nn.Sequential(
            ConvNeXtBlock(decoder_channels),
            ConvNeXtBlock(decoder_channels),
        )
        # uncertainty_head (e.g. {"head_type": "hetsngp", ...}) swaps the 1x1 conv
        # for a HetSNGP output layer (model/hetsngp.py) that returns per-pixel log
        # predictive probabilities -- valid logits, so the loss, argmax and the
        # upsampling below are unchanged. None keeps the exact plain classifier.
        if uncertainty_head:
            self.head = HetSNGPHead2d(decoder_channels, num_classes, **uncertainty_head)
        else:
            self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        # Separate attribute, never repurposing `.head` -- model/features.py's
        # pre-forward hook on `decoder.head` (uncertainty/, pc_align/,
        # mars-inference) must keep seeing exactly the DC classifier.
        self.head_ig = (
            nn.Conv2d(decoder_channels, num_classes_ig, kernel_size=1)
            if num_classes_ig
            else None
        )
        # Dropout2d(0) is the identity in both train and eval, so leaving this
        # permanently in the forward path (rather than branching on dropout>0)
        # costs nothing when the F4 lever is off and needs no special-casing
        # for resume/inference -- Dropout modules hold no state_dict entries.
        self.head_dropout = nn.Dropout2d(dropout)

    def forward(
        self,
        *,
        bottleneck: torch.Tensor,
        skip8: torch.Tensor,
        skip4: torch.Tensor,
        skip2: torch.Tensor,
        output_size: tuple[int, int],
        return_ig: bool = False,
    ):
        if self.aspp is not None:
            bottleneck = self.aspp(bottleneck)
        x = self.bottleneck_proj(bottleneck)

        x = F.interpolate(x, size=skip8.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse8(x + self.skip8_proj(skip8))

        x = F.interpolate(x, size=skip4.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse4(x + self.skip4_proj(skip4))

        x = F.interpolate(x, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse2(x + self.skip2_proj(skip2))

        # Classify at the decoder's own resolution, then upsample the logits --
        # NOT the other way round. A 1x1 conv mixes channels per pixel and
        # bilinear interpolation mixes pixels per channel, so the two commute
        # exactly (verified to 4e-15 in float64); doing the cheap operand first
        # avoids materialising a decoder_channels x H x W tensor at full input
        # resolution (16x more pixels here), which was ~15x slower and the
        # decoder's single largest activation.
        x = self.head_dropout(x)
        dc_logits = F.interpolate(
            self.head(x), size=output_size, mode="bilinear", align_corners=False
        )
        if isinstance(self.head, HetSNGPHead2d):
            # Bilinearly interpolated log-probabilities sum to <= 1 (Jensen), by up
            # to a few % at class boundaries; renormalize so the output stays an
            # exact log predictive. softmax/argmax/CE are invariant to this.
            dc_logits = F.log_softmax(dc_logits.float(), dim=1)
        if not return_ig:
            return dc_logits
        if self.head_ig is None:
            return dc_logits, None
        ig_logits = F.interpolate(
            self.head_ig(x), size=output_size, mode="bilinear", align_corners=False
        )
        return dc_logits, ig_logits


class SingleBranchSegmentationModel(nn.Module):
    """Segmentation head on the single-branch SimMIM ``HybridEncoder``.

    The encoder takes one image and returns a dict of multi-scale feature maps
    ``{s1: /4, s2: /8, s3: /16, s4: /32}``; those feed the same encoder-agnostic
    ``LightweightSegmentationDecoder`` used by the context model. The SimMIM
    reconstruction head is discarded -- only the pretrained encoder is reused.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_classes: int,
        bottleneck_channels: int,
        skip8_channels: int,
        skip4_channels: int,
        skip2_channels: int,
        decoder_channels: int = 256,
        num_classes_ig: Optional[int] = None,
        decoder_dropout: float = 0.0,
        use_aspp: bool = False,
        aspp_rates: Sequence[int] = (6, 12, 18),
        uncertainty_head: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = LightweightSegmentationDecoder(
            bottleneck_channels=bottleneck_channels,
            skip8_channels=skip8_channels,
            skip4_channels=skip4_channels,
            skip2_channels=skip2_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            num_classes_ig=num_classes_ig,
            dropout=decoder_dropout,
            use_aspp=use_aspp,
            aspp_rates=aspp_rates,
            uncertainty_head=uncertainty_head,
        )

    def forward(self, x: torch.Tensor, *, return_ig: bool = False):
        """Returns DC logits [B, num_classes, H, W] -- unchanged contract for
        every existing caller (mars-inference, uncertainty/, pc_align/,
        run_segmentation_epoch's normal path). `return_ig=True` (only ever
        passed internally by the Stage-3 training loop when the F1 aux head is
        active) instead returns `(dc_logits, ig_logits_or_None)`.
        """
        features = self.encoder(x)
        return self.decoder(
            bottleneck=features["s4"],
            skip8=features["s3"],
            skip4=features["s2"],
            skip2=features["s1"],
            output_size=x.shape[2:],
            return_ig=return_ig,
        )


class ContextAwareSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_classes: int,
        bottleneck_channels: int,
        skip8_channels: int,
        skip4_channels: int,
        skip2_channels: int,
        decoder_channels: int = 256,
        bottleneck_index: int = -1,
        skip8_index: int = 3,
        skip4_index: int = 1,
        skip2_index: int = 0,
        num_classes_ig: Optional[int] = None,
        decoder_dropout: float = 0.0,
        use_aspp: bool = False,
        aspp_rates: Sequence[int] = (6, 12, 18),
        use_context: bool = True,
        uncertainty_head: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = encoder
        # Mirrors the encoder's own toggle: with use_context=False this is a
        # single-branch model that happens to share the two-branch encoder family,
        # so a context on/off comparison differs ONLY in the context mechanism.
        self.use_context = use_context
        self.decoder = LightweightSegmentationDecoder(
            bottleneck_channels=bottleneck_channels,
            skip8_channels=skip8_channels,
            skip4_channels=skip4_channels,
            skip2_channels=skip2_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            num_classes_ig=num_classes_ig,
            dropout=decoder_dropout,
            use_aspp=use_aspp,
            aspp_rates=aspp_rates,
            uncertainty_head=uncertainty_head,
        )
        self.bottleneck_index = bottleneck_index
        self.skip8_index = skip8_index
        self.skip4_index = skip4_index
        self.skip2_index = skip2_index

    def forward(
        self,
        local_x: torch.Tensor,
        context_x: Optional[torch.Tensor] = None,
        *,
        return_ig: bool = False,
    ):
        """Same DC-logits-only contract as SingleBranchSegmentationModel.forward
        (see its docstring); `return_ig` behaves identically here."""
        if self.use_context and context_x is None:
            raise ValueError(
                "ContextAwareSegmentationModel was built with use_context=True but "
                "got context_x=None. Provide a context crop (see the context crop "
                "cache), or build the model with use_context=False."
            )

        features = (
            self.encoder(local_x, context_x)
            if self.use_context
            else self.encoder(local_x)
        )
        bottleneck = features[self.bottleneck_index]
        skip8 = features[self.skip8_index]
        skip4 = features[self.skip4_index]
        skip2 = features[self.skip2_index]

        return self.decoder(
            bottleneck=bottleneck,
            skip8=skip8,
            skip4=skip4,
            skip2=skip2,
            output_size=local_x.shape[2:],
            return_ig=return_ig,
        )
