from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channels for NCHW tensors.
    Input: [B, C, H, W]
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ------------------------------------------------------------
# Utility: stochastic depth
# ------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


# ------------------------------------------------------------
# Window helpers
# ------------------------------------------------------------
def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Partition NHWC feature map into non-overlapping windows.

    Input:
        x: [B, H, W, C]

    Output:
        windows: [B * num_windows, window_size, window_size, C]
    """
    b, h, w, c = x.shape

    x = x.view(
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, c)

    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Reverse window partition.

    Input:
        windows: [B * num_windows, window_size, window_size, C]

    Output:
        x: [B, H, W, C]
    """
    num_windows_h = h // window_size
    num_windows_w = w // window_size

    b = int(windows.shape[0] / (num_windows_h * num_windows_w))

    x = windows.view(
        b,
        num_windows_h,
        num_windows_w,
        window_size,
        window_size,
        -1,
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, -1)

    return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style residual block for NCHW feature maps.

    Structure:
        depthwise 7x7 conv
        LayerNorm2d
        pointwise conv expansion
        GELU
        pointwise conv projection
        layer scale
        residual connection
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )

        self.norm = LayerNorm2d(channels)

        self.pwconv1 = nn.Conv2d(
            channels,
            expansion * channels,
            kernel_size=1,
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(
            expansion * channels,
            channels,
            kernel_size=1,
        )

        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(channels),
            requires_grad=True,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.gamma[:, None, None] * x

        return residual + self.drop_path(x)
    



class ConvNeXtDownsample(nn.Module):
    """
    Learned downsampling block.

    Reduces spatial size by 2 and changes channel dimension.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNeXtStage(nn.Module):
    """
    Optional downsampling followed by several ConvNeXt blocks.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        if downsample:
            self.downsample = ConvNeXtDownsample(in_channels, out_channels)
        else:
            assert in_channels == out_channels
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(
                    out_channels,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x
    

class PatchMerging2D(nn.Module):
    """
    Simple CNN-style patch merging.

    In a full Swin implementation, patch merging is often done by concatenating
    neighboring 2x2 tokens and applying a linear layer. This Conv2d version is
    simpler for NCHW feature maps.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm = LayerNorm2d(in_channels)
        self.reduction = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Input:
        x: [B_windows, N, C]
           where N = window_size * window_size
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table.
        # Size: (2M-1) * (2M-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1),
                num_heads,
            )
        )

        # Pair-wise relative position index for each token inside a window.
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, M, M]

        coords_flatten = torch.flatten(coords, 1)  # [2, M*M]

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, M*M, M*M]

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coords.sum(-1)  # [M*M, M*M]

        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _relative_position_bias(self, dtype: torch.dtype) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()
        return relative_position_bias.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            [B_windows, N, C]
        mask:
            Optional attention mask for shifted windows.
            Shape: [num_windows, N, N]
        """
        b_windows, n, c = x.shape

        head_dim = c // self.num_heads
        qkv = self.qkv(x).reshape(
            b_windows,
            n,
            3,
            self.num_heads,
            head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_mask = self._relative_position_bias(dtype=q.dtype).unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            batch_size = b_windows // num_windows
            window_mask = mask.to(device=x.device, dtype=q.dtype).unsqueeze(1)
            attn_mask = (window_mask + attn_mask).repeat(batch_size, 1, 1, 1)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b_windows, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
    
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block for NCHW tensors.

    Alternates between:
      - normal window attention
      - shifted-window attention

    Input/output:
        [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        if shift_size >= window_size:
            raise ValueError("shift_size must be smaller than window_size.")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop=drop,
        )
        self._attn_mask_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}

    def _get_attention_mask(
        self,
        hp: int,
        wp: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create attention mask for shifted-window attention.
        Shape: [num_windows, M*M, M*M]
        """
        cache_key = (hp, wp, str(device), str(dtype))
        if cache_key in self._attn_mask_cache:
            return self._attn_mask_cache[cache_key]

        img_mask = torch.zeros(
            (1, hp, wp, 1),
            device=device,
            dtype=dtype,
        )

        m = self.window_size
        s = self.shift_size

        h_slices = (
            slice(0, -m),
            slice(-m, -s),
            slice(-s, None),
        )
        w_slices = (
            slice(0, -m),
            slice(-m, -s),
            slice(-s, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, m)
        mask_windows = mask_windows.view(-1, m * m)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(
            attn_mask != 0,
            float(-100.0),
        )
        attn_mask = attn_mask.masked_fill(
            attn_mask == 0,
            float(0.0),
        )

        self._attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape

        if c != self.dim:
            raise ValueError(
                f"Expected channel dim {self.dim}, got {c}."
            )

        # Keep the whole block in NHWC and convert back only once at the end.
        shortcut = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(shortcut)

        # Pad so H and W are divisible by window_size.
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        pad_r = (self.window_size - w % self.window_size) % self.window_size

        x = F.pad(
            x,
            (0, 0, 0, pad_r, 0, pad_b),
        )

        _, hp, wp, _ = x.shape

        # Shift.
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

            attn_mask = self._get_attention_mask(
                hp=hp,
                wp=wp,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows.
        x_windows = window_partition(
            shifted_x,
            self.window_size,
        )
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            c,
        )

        # Window attention.
        attn_windows = self.attn(
            x_windows,
            mask=attn_mask,
        )

        # Reverse windows.
        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            c,
        )

        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            hp,
            wp,
        )

        # Reverse shift.
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        # Remove padding.
        if pad_b > 0 or pad_r > 0:
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.permute(0, 3, 1, 2).contiguous()
    
    
class SwinStage(nn.Module):
    """
    Stack of Swin Transformer blocks.

    Input/output:
        [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        blocks = []

        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2

            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
    

class ConvNeXtSwinEncoder(nn.Module):
    """
    CNN/Swin hybrid encoder for HiRISE-style segmentation.

    Output feature pyramid:
      x1: 1/2
      x2: 1/4
      x3: 1/8
      x4: 1/8 after Swin
      x5: 1/16 after Swin
      x6: 1/32 after Swin, optional
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 48,
        use_stage32: bool = True,
        swin_depths: Sequence[int] = (2, 2, 2),
        swin_num_heads: Sequence[int] = (4, 8, 16),
        window_size: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()

        if len(swin_depths) != 3:
            raise ValueError("swin_depths must contain 3 stage depths.")
        if len(swin_num_heads) != 3:
            raise ValueError("swin_num_heads must contain 3 stage head counts.")

        c1 = base_channels          # 1/2
        c2 = base_channels * 2      # 1/4
        c3 = base_channels * 4      # 1/8
        c4 = base_channels * 8      # 1/16
        c5 = base_channels * 16     # 1/32

        # Stem: controlled first downsampling to 1/2 resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(c1),
            ConvNeXtBlock(c1),
        )

        # ConvNeXt local texture stages.
        self.stage4 = ConvNeXtStage(
            in_channels=c1,
            out_channels=c2,
            depth=2,
            downsample=True,
        )  # 1/4

        self.stage8 = ConvNeXtStage(
            in_channels=c2,
            out_channels=c3,
            depth=2,
            downsample=True,
        )  # 1/8

        # Swin contextual stages.
        self.swin8 = SwinStage(
            dim=c3,
            depth=swin_depths[0],
            num_heads=swin_num_heads[0],
            window_size=window_size,
            drop_path=drop_path,
        )  # still 1/8

        self.merge16 = PatchMerging2D(
            in_channels=c3,
            out_channels=c4,
        )

        self.swin16 = SwinStage(
            dim=c4,
            depth=swin_depths[1],
            num_heads=swin_num_heads[1],
            window_size=window_size,
            drop_path=drop_path,
        )  # 1/16

        self.use_stage32 = use_stage32

        if use_stage32:
            self.merge32 = PatchMerging2D(
                in_channels=c4,
                out_channels=c5,
            )
            self.swin32 = SwinStage(
                dim=c5,
                depth=swin_depths[2],
                num_heads=swin_num_heads[2],
                window_size=window_size,
                drop_path=drop_path,
            )

    def forward(self, x: torch.Tensor):
        x1 = self.stem(x)       # [B, c1, H/2,  W/2]
        x2 = self.stage4(x1)    # [B, c2, H/4,  W/4]
        x3 = self.stage8(x2)    # [B, c3, H/8,  W/8]

        x4 = self.swin8(x3)     # [B, c3, H/8,  W/8]

        x5 = self.merge16(x4)   # [B, c4, H/16, W/16]
        x5 = self.swin16(x5)

        if self.use_stage32:
            x6 = self.merge32(x5)   # [B, c5, H/32, W/32]
            x6 = self.swin32(x6)
            return x1, x2, x3, x4, x5, x6

        return x1, x2, x3, x4, x5

class LightweightContextEncoder(nn.Module):
    """
    Lightweight context branch.

    Input:
        context_x: [B, C, H, W]

    Intended use:
        context_x is a large surrounding crop, e.g. 2048×2048,
        already downsampled to the same tensor size as the local crop,
        e.g. 512×512.

    Output:
        context_vector: [B, context_dim]

    This branch is intentionally small. It should provide scene-level context,
    not do full segmentation.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 24,
        depth_per_stage: int = 1,
        context_dim: int = 256,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(c1),
            ConvNeXtBlock(c1),
        )  # 1/2

        self.stage4 = ConvNeXtStage(
            in_channels=c1,
            out_channels=c2,
            depth=depth_per_stage,
            downsample=True,
        )  # 1/4

        self.stage8 = ConvNeXtStage(
            in_channels=c2,
            out_channels=c3,
            depth=depth_per_stage,
            downsample=True,
        )  # 1/8

        self.stage16 = ConvNeXtStage(
            in_channels=c3,
            out_channels=c4,
            depth=depth_per_stage,
            downsample=True,
        )  # 1/16

        self.stage32 = ConvNeXtStage(
            in_channels=c4,
            out_channels=c5,
            depth=depth_per_stage,
            downsample=True,
        )  # 1/32

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c5, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

    def forward(self, context_x: torch.Tensor) -> torch.Tensor:
        x = self.stem(context_x)
        x = self.stage4(x)
        x = self.stage8(x)
        x = self.stage16(x)
        x = self.stage32(x)

        context_vector = self.proj(self.pool(x))

        return context_vector
    
class ContextFiLM2d(nn.Module):
    """
    FiLM-style context conditioning for NCHW feature maps.

    Given:
        feature:        [B, C, H, W]
        context_vector: [B, D]

    Applies:
        feature * (1 + gamma) + beta

    The final linear layer is initialized to zero so the module starts
    as an identity transform.
    """

    def __init__(
        self,
        feature_channels: int,
        context_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.feature_channels = feature_channels

        self.to_scale_shift = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_channels),
        )

        # Start as identity: scale = 0, shift = 0.
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        context_vector: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift = self.to_scale_shift(context_vector)

        scale, shift = scale_shift.chunk(2, dim=1)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        return feature * (1.0 + scale) + shift
    

class ContextAwareConvNeXtSwinEncoder(nn.Module):
    """
    Two-branch encoder:

    Local branch:
        native-resolution crop, e.g. 512×512

    Context branch:
        larger surrounding crop downsampled to 512×512

    The context branch does not segment. It produces a global context vector
    that modulates selected local feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        local_base_channels: int = 48,
        context_base_channels: int = 24,
        context_dim: int = 256,
        use_stage32: bool = True,
        swin_depths: Sequence[int] = (2, 2, 2),
        swin_num_heads: Sequence[int] = (4, 8, 16),
        window_size: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.use_stage32 = use_stage32

        self.local_encoder = ConvNeXtSwinEncoder(
            in_channels=in_channels,
            base_channels=local_base_channels,
            use_stage32=use_stage32,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            window_size=window_size,
            drop_path=drop_path,
        )

        self.context_encoder = LightweightContextEncoder(
            in_channels=in_channels,
            base_channels=context_base_channels,
            depth_per_stage=1,
            context_dim=context_dim,
        )

        c1 = local_base_channels          # 1/2
        c2 = local_base_channels * 2      # 1/4
        c3 = local_base_channels * 4      # 1/8
        c4 = local_base_channels * 8      # 1/16
        c5 = local_base_channels * 16     # 1/32

        # I would not condition the very earliest x1 feature at first.
        # It may inject context into very local texture too aggressively.
        self.film_x2 = ContextFiLM2d(c2, context_dim)
        self.film_x3 = ContextFiLM2d(c3, context_dim)
        self.film_x4 = ContextFiLM2d(c4, context_dim)
        self.film_x5 = ContextFiLM2d(c5, context_dim)

        if use_stage32:
            self.film_x6 = ContextFiLM2d(c5, context_dim)

    def forward(
        self,
        local_x: torch.Tensor,
        context_x: torch.Tensor,
    ):
        """
        Parameters
        ----------
        local_x:
            Native-resolution local crop.
            Example: [B, C, 512, 512]

        context_x:
            Larger surrounding crop already downsampled.
            Example: original 2048×2048 context crop resized to [B, C, 512, 512]

        Returns
        -------
        Feature pyramid for the decoder.
        """

        context_vector = self.context_encoder(context_x)

        features = self.local_encoder(local_x)

        if self.use_stage32:
            x1, x2, x3, x4, x5, x6 = features

            x2 = self.film_x2(x2, context_vector)
            x3 = self.film_x3(x3, context_vector)
            x4 = self.film_x4(x4, context_vector)
            x5 = self.film_x5(x5, context_vector)
            x6 = self.film_x6(x6, context_vector)

            return x1, x2, x3, x4, x5, x6

        x1, x2, x3, x4, x5 = features

        x2 = self.film_x2(x2, context_vector)
        x3 = self.film_x3(x3, context_vector)
        x4 = self.film_x4(x4, context_vector)
        x5 = self.film_x5(x5, context_vector)

        return x1, x2, x3, x4, x5

class PatchMasker(nn.Module):
    """
    Random patch masker for image-like tensors.

    Input:
        x: [B, C, H, W]

    Output:
        x_masked: [B, C, H, W]
        mask:     [B, 1, H, W], where 1 means "masked / reconstruct this"
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 16,
        mask_ratio: float = 0.6,
    ):
        super().__init__()

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1.")

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # One learnable value per input channel.
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}. "
                f"Got H={h}, W={w}."
            )

        hp = h // self.patch_size
        wp = w // self.patch_size

        # Patch-level mask: [B, 1, H_patch, W_patch]
        patch_mask = torch.rand(
            b,
            1,
            hp,
            wp,
            device=x.device,
            dtype=x.dtype,
        ) < self.mask_ratio

        patch_mask = patch_mask.to(dtype=x.dtype)

        # Upsample to pixel-level mask: [B, 1, H, W]
        mask = F.interpolate(
            patch_mask,
            size=(h, w),
            mode="nearest",
        )

        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.make_mask(x)

        # Replace masked pixels with learnable mask token.
        x_masked = x * (1.0 - mask) + self.mask_token * mask

        return x_masked, mask
    

class WeakReconstructionDecoder(nn.Module):
    """
    Weak decoder for masked reconstruction pretraining.

    It is intentionally weaker than a segmentation decoder.

    It can use:
        - bottleneck feature only
        - bottleneck + one low-resolution skip, usually 1/8

    It should NOT use high-resolution 1/2 or 1/4 skips during pretraining.
    """
    def __init__(
        self,
        bottleneck_channels: int,
        out_channels: int,
        decoder_channels: int = 256,
        skip8_channels: Optional[int] = None,
        use_skip8: bool = True,
    ):
        super().__init__()

        self.use_skip8 = use_skip8 and skip8_channels is not None

        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                decoder_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            ConvNeXtBlock(decoder_channels),
        )

        if self.use_skip8:
            self.skip8_proj = nn.Sequential(
                nn.Conv2d(
                    skip8_channels,
                    decoder_channels,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_channels),
                nn.GELU(),
            )

            self.fuse = nn.Sequential(
                ConvNeXtBlock(decoder_channels),
                ConvNeXtBlock(decoder_channels),
            )
        else:
            self.skip8_proj = None
            self.fuse = nn.Sequential(
                ConvNeXtBlock(decoder_channels),
            )

        # 1x1 head keeps the full-resolution reconstruction part weak.
        self.reconstruction_head = nn.Conv2d(
            decoder_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        output_size: Tuple[int, int],
        skip8: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.bottleneck_proj(bottleneck)

        if self.use_skip8 and skip8 is not None:
            # Bring bottleneck to 1/8 resolution.
            x = F.interpolate(
                x,
                size=skip8.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            skip = self.skip8_proj(skip8)

            # Additive fusion is weaker than concatenation and avoids an overly
            # powerful reconstruction path.
            x = x + skip
            x = self.fuse(x)

        # Directly upsample to full resolution.
        # This is intentionally simple.
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        reconstruction = self.reconstruction_head(x)

        return reconstruction
    


class MaskedReconstructionPretrainer(nn.Module):
    """
    Self-supervised masked reconstruction wrapper.

    This module:
        1. masks input patches
        2. runs the masked image through the encoder
        3. reconstructs the original input
        4. computes loss only on masked pixels

    Intended usage:
        - train encoder + reconstruction decoder
        - discard reconstruction decoder
        - reuse encoder for supervised segmentation
    """
    def __init__(
        self,
        encoder: nn.Module,
        in_channels: int,
        bottleneck_channels: int,
        decoder_channels: int = 256,
        patch_size: int = 16,
        mask_ratio: float = 0.6,
        bottleneck_index: int = -1,
        skip8_index: Optional[int] = None,
        skip8_channels: Optional[int] = None,
        use_skip8: bool = True,
        loss_type: str = "l1",
    ):
        super().__init__()

        self.encoder = encoder

        self.masker = PatchMasker(
            in_channels=in_channels,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
        )

        self.decoder = WeakReconstructionDecoder(
            bottleneck_channels=bottleneck_channels,
            out_channels=in_channels,
            decoder_channels=decoder_channels,
            skip8_channels=skip8_channels,
            use_skip8=use_skip8,
        )

        self.bottleneck_index = bottleneck_index
        self.skip8_index = skip8_index
        self.loss_type = loss_type

        if loss_type not in {"l1", "mse", "smooth_l1"}:
            raise ValueError(
                "loss_type must be one of: 'l1', 'mse', 'smooth_l1'."
            )

    def reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked pixels.

        reconstruction: [B, C, H, W]
        target:         [B, C, H, W]
        mask:           [B, 1, H, W]
        """

        # Broadcast mask from [B,1,H,W] to [B,C,H,W].
        mask = mask.expand_as(target)

        if self.loss_type == "l1":
            loss_map = torch.abs(reconstruction - target)
        elif self.loss_type == "mse":
            loss_map = (reconstruction - target).pow(2)
        elif self.loss_type == "smooth_l1":
            loss_map = F.smooth_l1_loss(
                reconstruction,
                target,
                reduction="none",
            )
        else:
            raise RuntimeError("Invalid loss type.")

        # Avoid division by zero, although mask should never be empty.
        denom = mask.sum().clamp_min(1.0)

        loss = (loss_map * mask).sum() / denom

        return loss

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining.

        Returns a dictionary so you can log reconstruction, mask, etc.
        """
        input_size = x.shape[2:]

        x_masked, mask = self.masker(x)

        features = self.encoder(x_masked)

        if not isinstance(features, (tuple, list)):
            raise TypeError(
                "Encoder must return a tuple/list of feature maps."
            )

        bottleneck = features[self.bottleneck_index]

        skip8 = None
        if self.skip8_index is not None:
            skip8 = features[self.skip8_index]

        reconstruction = self.decoder(
            bottleneck=bottleneck,
            output_size=input_size,
            skip8=skip8,
        )

        loss = self.reconstruction_loss(
            reconstruction=reconstruction,
            target=x,
            mask=mask,
        )

        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "masked_input": x_masked,
            "mask": mask,
        }


class ContextAwareMaskedReconstructionPretrainer(MaskedReconstructionPretrainer):
    """
    Context-aware masked reconstruction wrapper.

    This variant masks only the local crop, then reconstructs that local crop
    while conditioning the encoder on a separate, larger context crop.
    """

    def forward(
        self,
        local_x: torch.Tensor,
        context_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        local_x:
            Local crop to be masked and reconstructed.
        context_x:
            Larger context crop that remains unmasked and conditions the local
            encoder path.
        """
        input_size = local_x.shape[2:]

        local_masked, mask = self.masker(local_x)

        features = self.encoder(local_masked, context_x)

        if not isinstance(features, (tuple, list)):
            raise TypeError(
                "Encoder must return a tuple/list of feature maps."
            )

        bottleneck = features[self.bottleneck_index]

        skip8 = None
        if self.skip8_index is not None:
            skip8 = features[self.skip8_index]

        reconstruction = self.decoder(
            bottleneck=bottleneck,
            output_size=input_size,
            skip8=skip8,
        )

        loss = self.reconstruction_loss(
            reconstruction=reconstruction,
            target=local_x,
            mask=mask,
        )

        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "masked_input": local_masked,
            "mask": mask,
            "context_input": context_x,
        }
