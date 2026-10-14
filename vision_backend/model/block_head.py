"""Block-level classifier: one class per 512x512 block, on a trained segmenter.

The planetary scientists label whole blocks with one class -- one of the 14
NOAH-H DC classes or one of the new classes the segmenter has never seen. The
block logit for class k is

    logit_k = alpha * prior_k + b_k + w_k . [LN(z), pbar]

- prior_k: for a NOAH-H class, log of the segmenter's mean predicted
  probability of k over the block's valid pixels; 0 for a new class.
- pbar: those mean probabilities themselves (the block's predicted class
  composition), so a new class can be learned as a mixture of known ones.
- z: gated-attention pooling (Ilse, Tomczak & Welling, "Attention-based Deep
  Multiple Instance Learning", ICML 2018) of the decoder features that the
  segmentation head classifies, at 1/4 resolution -- a learned, mask-aware
  weighting of the block's pixels, so a class defined by a small conspicuous
  feature (e.g. a boulder field) is not averaged away the way the mean is.

Initialised with alpha = 1, w = 0 and b = 0 for NOAH-H classes, the classifier
*is* the segmenter's dominant-class rule (argmax of mean probability) before
any block training: epoch 0 is a zero-shot baseline on the NOAH-H classes, and
fine-tuning starts from it instead of from a random head. New classes start at
b = new_class_init_logit (default log(1/K)) and are learned through w.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool2d(nn.Module):
    """Gated attention-MIL pooling (Ilse et al. 2018, eq. 9) over a feature map.

    a_i = softmax_i( w^T (tanh(V h_i) * sigmoid(U h_i)) + log m_i ),  z = sum_i a_i h_i

    `weight` m_i in [0, 1] is the fraction of valid (non-nodata) input pixels
    under feature position i; adding log m_i to the score is the soft version
    of masking nodata out of the softmax.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn_v = nn.Conv2d(dim, hidden, kernel_size=1)
        self.attn_u = nn.Conv2d(dim, hidden, kernel_size=1)
        self.attn_w = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        score = self.attn_w(torch.tanh(self.attn_v(x)) * torch.sigmoid(self.attn_u(x)))
        score = score.float().flatten(1) + torch.log(weight.float().flatten(1).clamp_min(1e-6))
        attn = torch.softmax(score, dim=1)  # [B, h*w]
        return torch.einsum("bn,bcn->bc", attn, x.float().flatten(2))


class BlockClassifier(nn.Module):
    """Wraps a `SingleBranchSegmentationModel`; returns block logits [B, K].

    `seg_index[k]` is the segmenter's DC class index for block class k, or None
    for a new class. Every segmenter class should appear in `seg_index` (classes
    absent from the block labels still shape the prior's normalisation).
    """

    def __init__(
        self,
        seg_model: nn.Module,
        seg_index: Sequence[Optional[int]],
        *,
        attn_hidden: int = 128,
        dropout: float = 0.1,
        new_class_init_logit: Optional[float] = None,
        tile_size: Optional[int] = 512,
    ):
        super().__init__()
        # Blocks larger than tile_size (e.g. 1024 expert blocks) go through the
        # backbone as tile_size tiles -- the input size it was trained on, 0.25 m
        # GSD kept -- and the tiles' decoder features are stitched back into one
        # map, so prior and attention pooling still cover the whole labelled block.
        self.tile_size = tile_size
        head = getattr(getattr(seg_model, "decoder", None), "head", None)
        if not isinstance(head, nn.Conv2d) or not hasattr(seg_model, "decode_features"):
            raise TypeError(
                "BlockClassifier needs a single-branch segmentation model with a plain "
                "Conv2d DC head (model_kind 'simmim', no uncertainty head)."
            )
        num_seg = int(head.out_channels)
        num_classes = len(seg_index)
        seg_ids = [int(s) for s in seg_index if s is not None]
        if len(set(seg_ids)) != len(seg_ids) or any(not 0 <= s < num_seg for s in seg_ids):
            raise ValueError(f"seg_index must map to distinct segmenter classes in [0, {num_seg}): {seg_index}")

        self.seg = seg_model
        self.num_classes = num_classes
        self.num_seg_classes = num_seg
        self.seg_index = [None if s is None else int(s) for s in seg_index]
        # prior[:, k] = log_pbar[:, prior_src[k]] * prior_mask[k] -- indexing, not a
        # matmul, so bf16 autocast never rounds the log-probabilities (which could
        # flip near-ties of the zero-shot dominant-class rule).
        self.register_buffer(
            "prior_src", torch.tensor([s or 0 for s in self.seg_index], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "prior_mask", torch.tensor([float(s is not None) for s in self.seg_index]), persistent=False
        )

        dim = int(head.in_channels)
        self.pool = GatedAttentionPool2d(dim, attn_hidden)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim + num_seg, num_classes)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.prior_scale = nn.Parameter(torch.ones(()))
        init = math.log(1.0 / num_classes) if new_class_init_logit is None else float(new_class_init_logit)
        self.class_bias = nn.Parameter(
            torch.tensor([0.0 if s is not None else init for s in self.seg_index])
        )

    def head_parameters(self):
        """Parameters of the new block head (everything except the segmenter)."""
        return [p for n, p in self.named_parameters() if not n.startswith("seg.")]

    def decode_features(self, image: torch.Tensor) -> torch.Tensor:
        t = self.tile_size
        b, c, h, w = image.shape
        if t is None or (h <= t and w <= t):
            return self.seg.decode_features(image)
        if h % t or w % t:
            raise ValueError(f"block {h}x{w} is not a multiple of tile_size {t}")
        nh, nw = h // t, w // t
        tiles = image.reshape(b, c, nh, t, nw, t).permute(0, 2, 4, 1, 3, 5).reshape(b * nh * nw, c, t, t)
        f = self.seg.decode_features(tiles)
        fc, fh, fw = f.shape[1:]
        return f.reshape(b, nh, nw, fc, fh, fw).permute(0, 3, 1, 4, 2, 5).reshape(b, fc, nh * fh, nw * fw)

    def forward(self, image: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """image [B, 1, H, W]; valid [B, 1, H, W] (1 = imagery, 0 = nodata) or None."""
        phi, offset = self.block_features(image, valid)
        return offset + self.fc(phi)

    def block_features(self, image: torch.Tensor, valid: Optional[torch.Tensor] = None):
        """(phi, offset) with logits = offset + fc(phi).

        phi = [LN(z), p_bar] [B, C + S] is the input of the last layer `fc` --
        the block feature for last-layer Laplace, Mahalanobis scoring and neural
        PCA (psi_k = fc.weight[k] * phi). offset = alpha * prior + class_bias
        [B, K] is the part of the logits that `fc` does not see.
        """
        feats = self.decode_features(image)                      # [B, C, h, w]
        pix = self.seg.decoder.head(self.seg.decoder.head_dropout(feats))
        if valid is None:
            weight = torch.ones_like(pix[:, :1], dtype=torch.float32)
        else:
            # fraction of valid input pixels under each feature position
            weight = F.adaptive_avg_pool2d(valid.float(), pix.shape[2:])

        log_p = F.log_softmax(pix.float(), dim=1)                # [B, S, h, w]
        log_w = torch.log(weight.clamp_min(1e-6))
        total = weight.flatten(1).sum(1).clamp_min(1e-6)        # [B]
        # log of the valid-weighted mean probability, computed in log space
        log_pbar = torch.logsumexp((log_p + log_w).flatten(2), dim=2) - torch.log(total)[:, None]

        z = self.dropout(self.norm(self.pool(feats, weight)))
        prior = log_pbar[:, self.prior_src] * self.prior_mask    # [B, K], 0 for new classes
        return torch.cat([z, log_pbar.exp()], dim=1), self.prior_scale * prior + self.class_bias
