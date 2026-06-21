"""Relative-position-bias utilities for resolution transfer.

The windowed Swin blocks use a fixed window size (8), so their relative
position bias table is independent of the input resolution and never needs
interpolation. The global attention block in stage S4 is different: its
"window" is the entire S4 grid (32x32 at 1024 input, 64x64 at 2048, ...), so
its bias table must be resized when the grid changes between training and
inference. This module provides that bicubic interpolation plus the helper
that builds the pairwise relative-position index for a square grid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_relative_position_index(grid: int) -> torch.Tensor:
    """Pairwise relative-position index for a ``grid x grid`` token map.

    Returns a ``(grid*grid, grid*grid)`` long tensor whose entries index into a
    bias table of length ``(2*grid - 1) ** 2`` (the standard Swin layout).
    """
    coords_h = torch.arange(grid)
    coords_w = torch.arange(grid)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, G, G]
    coords_flatten = torch.flatten(coords, 1)  # [2, G*G]

    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
    relative_coords[:, :, 0] += grid - 1
    relative_coords[:, :, 1] += grid - 1
    relative_coords[:, :, 0] *= 2 * grid - 1

    return relative_coords.sum(-1)  # [N, N]


def interpolate_relative_position_bias(
    table: torch.Tensor,
    src_grid: int,
    dst_grid: int,
    num_heads: int,
) -> torch.Tensor:
    """Bicubic-resize a relative-position-bias table to a new grid size.

    Parameters
    ----------
    table:
        ``((2*src_grid - 1) ** 2, num_heads)`` bias table for a square grid.
    src_grid, dst_grid:
        Source and target square grid side lengths.
    num_heads:
        Number of attention heads (table's second dim).

    Returns
    -------
    ``((2*dst_grid - 1) ** 2, num_heads)`` table for the target grid. When the
    grids match, the input table is returned unchanged.
    """
    if src_grid == dst_grid:
        return table

    src_side = 2 * src_grid - 1
    dst_side = 2 * dst_grid - 1

    # [L, heads] -> [1, heads, src_side, src_side]
    table = table.reshape(src_side, src_side, num_heads)
    table = table.permute(2, 0, 1).unsqueeze(0)

    table = F.interpolate(
        table.float(),
        size=(dst_side, dst_side),
        mode="bicubic",
        align_corners=False,
    )

    # back to [(2*dst-1)**2, heads]
    table = table.squeeze(0).permute(1, 2, 0).reshape(dst_side * dst_side, num_heads)
    return table
