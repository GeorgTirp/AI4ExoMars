from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from vision_backend.model.model import ConvNeXtBlock, LayerNorm2d
except ModuleNotFoundError:
    from model.model import ConvNeXtBlock, LayerNorm2d


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
    ):
        super().__init__()

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
        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(
        self,
        *,
        bottleneck: torch.Tensor,
        skip8: torch.Tensor,
        skip4: torch.Tensor,
        skip2: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x = self.bottleneck_proj(bottleneck)

        x = F.interpolate(x, size=skip8.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse8(x + self.skip8_proj(skip8))

        x = F.interpolate(x, size=skip4.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse4(x + self.skip4_proj(skip4))

        x = F.interpolate(x, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse2(x + self.skip2_proj(skip2))

        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.head(x)


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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(
            bottleneck=features["s4"],
            skip8=features["s3"],
            skip4=features["s2"],
            skip2=features["s1"],
            output_size=x.shape[2:],
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
        )
        self.bottleneck_index = bottleneck_index
        self.skip8_index = skip8_index
        self.skip4_index = skip4_index
        self.skip2_index = skip2_index

    def forward(
        self,
        local_x: torch.Tensor,
        context_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_x is None:
            raise ValueError(
                "ContextAwareSegmentationModel expects both local_x and context_x."
            )

        features = self.encoder(local_x, context_x)
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
        )
