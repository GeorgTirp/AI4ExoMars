"""Single-branch ConvNeXt-V2 / Swin hybrid encoder (Model v2, frozen contract).

Stage layout (input (B, 1, H, W), H and W divisible by 256):

    stem   4x4 conv stride 4      stride 4    dim 96    depth -
    S1     ConvNeXtV2Block x3     stride 4    dim 96    depth 3
    S2     ConvNeXtV2Block x3     stride 8    dim 192   depth 3
    S3     Swin x6 (12 heads)     stride 16   dim 384   depth 6
    S4     Swin x1 (24 heads)
           + GlobalAttention x1   stride 32   dim 768   depth 2

Downsample between stages: LayerNorm2d + 2x2 stride-2 conv. Stochastic depth
ramps linearly 0 -> 0.1 across all 14 blocks. forward returns a dict with
``s1, s2, s3, s4`` feature maps (s4 post final LayerNorm) and ``pooled``
(global-average-pooled s4).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks_v2 import ConvNeXtV2Block, Downsample2x, GlobalAttentionBlock
from .model import LayerNorm2d, SwinTransformerBlock

# Frozen stage configuration. Do not change without sign-off (§2 / §10).
STEM_STRIDE = 4
STAGE_DIMS = (96, 192, 384, 768)
STAGE_DEPTHS = (3, 3, 6, 2)
S3_HEADS = 12
S4_HEADS = 24
WINDOW_SIZE = 8
MAX_DROP_PATH = 0.1


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        *,
        global_base_grid: int = 32,
        window_size: int = WINDOW_SIZE,
        drop_path: float = MAX_DROP_PATH,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        d1, d2, d3, d4 = STAGE_DIMS
        n1, n2, n3, n4 = STAGE_DEPTHS
        total_blocks = n1 + n2 + n3 + n4  # 14
        dp_rates = torch.linspace(0.0, drop_path, total_blocks).tolist()

        # stem: 4x4 conv stride 4 (patchify), 1 -> 96
        self.stem = nn.Conv2d(in_channels, d1, kernel_size=STEM_STRIDE, stride=STEM_STRIDE)

        i = 0
        # S1: ConvNeXt-V2, stride 4
        self.s1 = nn.ModuleList(
            [ConvNeXtV2Block(d1, drop_path=dp_rates[i + k]) for k in range(n1)]
        )
        i += n1

        self.down1 = Downsample2x(d1, d2)
        # S2: ConvNeXt-V2, stride 8
        self.s2 = nn.ModuleList(
            [ConvNeXtV2Block(d2, drop_path=dp_rates[i + k]) for k in range(n2)]
        )
        i += n2

        self.down2 = Downsample2x(d2, d3)
        # S3: Swin, stride 16, 12 heads, alternating shift
        self.s3 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=d3,
                    num_heads=S3_HEADS,
                    window_size=window_size,
                    shift_size=0 if k % 2 == 0 else window_size // 2,
                    drop_path=dp_rates[i + k],
                )
                for k in range(n3)
            ]
        )
        i += n3

        self.down3 = Downsample2x(d3, d4)
        # S4: 1 Swin (W-MSA) + 1 global attention, stride 32, 24 heads
        self.s4 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=d4,
                    num_heads=S4_HEADS,
                    window_size=window_size,
                    shift_size=0,
                    drop_path=dp_rates[i],
                ),
                GlobalAttentionBlock(
                    dim=d4,
                    num_heads=S4_HEADS,
                    base_grid=global_base_grid,
                    drop_path=dp_rates[i + 1],
                ),
            ]
        )

        self.norm = LayerNorm2d(d4)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D (B, C, H, W) tensor, got shape {tuple(x.shape)}.")
        _, _, h, w = x.shape
        if h % 256 != 0 or w % 256 != 0:
            raise ValueError(
                f"Input H and W must be divisible by 256 (stride 32 x window 8); got {h}x{w}."
            )

        x = self.stem(x)
        s1 = self._run(self.s1, x)             # (B, 96,  H/4,  W/4)
        s2 = self._run(self.s2, self.down1(s1))  # (B, 192, H/8,  W/8)
        s3 = self._run(self.s3, self.down2(s2))  # (B, 384, H/16, W/16)
        # S4 (global block is never checkpointed; it holds the rel-pos transfer)
        x4 = self.down3(s3)
        for block in self.s4:
            if self.use_checkpoint and self.training and isinstance(block, SwinTransformerBlock):
                x4 = checkpoint(block, x4, use_reentrant=False)
            else:
                x4 = block(x4)
        s4 = self.norm(x4)                     # (B, 768, H/32, W/32)
        pooled = s4.mean(dim=(2, 3))           # (B, 768)

        return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "pooled": pooled}

    # ---- introspection -------------------------------------------------
    def param_table(self) -> List[tuple[str, int]]:
        """Per-component parameter counts for the startup table / CI check."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        downs = count(self.down1) + count(self.down2) + count(self.down3)
        rows = [
            ("stem", count(self.stem)),
            ("S1", count(self.s1)),
            ("S2", count(self.s2)),
            ("S3", count(self.s3)),
            ("S4", count(self.s4) + count(self.norm)),
            ("downsample", downs),
        ]
        rows.append(("encoder total", sum(p.numel() for p in self.parameters())))
        return rows
