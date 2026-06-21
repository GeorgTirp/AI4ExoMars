"""Building blocks for the v2 single-branch hybrid encoder.

These complement the blocks already in ``model.model`` (which are reused
as-is where correct):

- ``GRN``                 global response normalization (ConvNeXt-V2).
- ``ConvNeXtV2Block``     ConvNeXt-V2 residual block (GRN, no layer scale).
- ``Downsample2x``        LayerNorm + 2x2 stride-2 conv between stages.
- ``GlobalAttentionBlock``full self-attention over the whole S4 grid, with a
                          resolution-transferable relative position bias.

The windowed Swin blocks (``SwinTransformerBlock`` / ``WindowAttention``) and
``LayerNorm2d`` / ``DropPath`` / ``MLP`` are imported from ``model.model``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DropPath, LayerNorm2d, MLP
from .relpos import build_relative_position_index, interpolate_relative_position_bias


class GRN(nn.Module):
    """Global Response Normalization for NCHW tensors (ConvNeXt-V2).

    Normalizes each channel's global (spatial) L2 response by the mean response
    across channels, then applies a learnable per-channel affine with a residual.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global spatial response per channel
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 residual block (NCHW).

    7x7 DW conv -> LayerNorm2d -> 1x1 expand (4x) -> GELU -> GRN -> 1x1 project
    -> DropPath -> residual. No layer scale (GRN replaces it).
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = expansion * channels

        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


class Downsample2x(nn.Module):
    """Between-stage downsampler: LayerNorm2d + 2x2 stride-2 conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(self.norm(x))


class GlobalAttentionBlock(nn.Module):
    """Pre-LN full self-attention over all tokens of a square grid + MLP.

    Holds a relative-position-bias table sized for ``base_grid`` (the training
    S4 grid). At a different (square) grid it bicubically interpolates the table
    so the same weights transfer across inference resolutions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        base_grid: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.base_grid = base_grid
        self.scale = (dim // num_heads) ** -0.5
        self.attn_drop_p = attn_drop

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * base_grid - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.register_buffer(
            "_base_index",
            build_relative_position_index(base_grid),
            persistent=False,
        )
        # grid -> relative_position_index, built lazily for non-base grids
        self._index_cache: Dict[int, torch.Tensor] = {}

    def _relative_position_index(self, grid: int, device: torch.device) -> torch.Tensor:
        if grid == self.base_grid:
            return self._base_index.to(device)
        index = self._index_cache.get(grid)
        if index is None or index.device != device:
            index = build_relative_position_index(grid).to(device)
            self._index_cache[grid] = index
        return index

    def _relative_position_bias(self, grid: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        table = self.relative_position_bias_table
        if grid != self.base_grid:
            table = interpolate_relative_position_bias(
                table, self.base_grid, grid, self.num_heads
            )
        index = self._relative_position_index(grid, device)
        bias = table[index.reshape(-1)]  # [N*N, heads]
        n = grid * grid
        bias = bias.view(n, n, self.num_heads).permute(2, 0, 1).contiguous()
        return bias.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != w:
            raise ValueError(
                f"GlobalAttentionBlock requires a square grid, got {h}x{w}."
            )
        grid = h
        n = h * w

        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        shortcut = tokens
        y = self.norm1(tokens)

        head_dim = c // self.num_heads
        qkv = self.qkv(y).reshape(b, n, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        bias = self._relative_position_bias(grid, q.dtype, q.device).unsqueeze(0)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(b, n, c)
        attn = self.proj_drop(self.proj(attn))

        tokens = shortcut + self.drop_path(attn)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))

        return tokens.transpose(1, 2).reshape(b, c, h, w)
