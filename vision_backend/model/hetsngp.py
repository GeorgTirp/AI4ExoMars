"""Heteroscedastic SNGP (HetSNGP) output layer for dense prediction.

Implementation source
---------------------
Fortuin, Collier, Wenzel, Allingham, Liu, Tran, Lakshminarayanan, Berent,
Jenatton, Kokiopoulou. "Deep Classifiers with Label Noise Modeling and Distance
Awareness." TMLR 2022, arXiv:2110.02609 -- Eqs. (4)-(6), Algorithms 1-2 and the
Laplace derivation of Appendix A.2. Details the paper leaves open were matched
to the authors' reference layer, edward2 ``HeteroscedasticSNGPLayer``
(tensorflow/layers/hetsngp.py), which composes ``RandomFeatureGaussianProcess``
(SNGP: Liu et al., NeurIPS 2020) and ``MCSoftmaxDenseFA`` (Collier et al.,
CVPR 2021):

- GP input LayerNorm, orthogonal random features (stddev 1), bias ~ U(0, 2pi),
  features scaled by sqrt(2/m), cos activation, input scaled 1/sqrt(kernel_scale);
- heteroscedastic diagonal = softplus(.) + 1e-3, low-rank factor = affine(h),
  both computed from the same representation h that feeds the GP;
- returned "logits" are log(mean_s softmax(u_s / tau)), so the ordinary
  cross-entropy on them is the HetSNGP negative log-likelihood;
- at prediction the GP marginal variance is added to the diagonal of the noise
  covariance, i.e. scale_c = sqrt(d_c^2 + Phi^T Sigma_c Phi).

Model, with every decoder pixel a data point (as in per-pixel heteroscedastic
segmentation, Kendall & Gal 2017):

    Phi(h) = sqrt(2/m) cos(W LN(h) / sqrt(kernel_scale) + b)
    u_c    = Phi^T beta_c + d_c(h) eps_K,c + [V(h) eps_R]_c          (Eq. 4)
    p(y=c) = 1/S sum_s softmax(u_s / tau)_c                          (Eq. 6)

Training (Alg. 1) keeps beta at its MAP value and samples only the
heteroscedastic noise. Prediction (Alg. 2) also samples beta_c from the Laplace
posterior N(beta_hat_c, Sigma_c), Sigma_c^-1 = I + sum_i p_ic (1 - p_ic) Phi_i
Phi_i^T (Eq. 5). The paper accumulates that sum during the final epoch; here
`fit_laplace_covariance` computes it after training, on the final (EMA) weights
that are actually saved and used for prediction, so the Laplace expansion sits
at the weights it is meant to describe.

Spectral normalization (the "SN" in SNGP) is not applied: the paper found it
unnecessary for distance awareness with a vision transformer backbone and ran
those experiments without it (Sec. 5.4); this decoder sits on the
ConvNeXt-V2/Swin/global-attention HybridEncoder.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LayerNorm2d

MIN_SCALE_MONTE_CARLO = 1e-3  # edward2 heteroscedastic.py
HEAD_TYPES = ("het", "sngp", "hetsngp")


def orthogonal_random_features(num_features: int, in_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Orthogonal random features (Yu et al. 2016), stddev 1, shape [num_features, in_dim].

    Rows come in blocks of an orthogonal matrix, each row rescaled to the norm of
    an independent standard Gaussian vector, so every row is marginally
    N(0, I) but rows within a block are orthogonal (lower-variance kernel
    estimate than i.i.d. Gaussian features). Matches edward2's default
    ``OrthogonalRandomFeatures(stddev=1.0)``.
    """
    blocks = []
    for _ in range(math.ceil(num_features / in_dim)):
        q, _ = torch.linalg.qr(torch.randn(in_dim, in_dim, generator=generator))
        norms = torch.randn(in_dim, in_dim, generator=generator).norm(dim=1)
        blocks.append(q * norms[:, None])
    return torch.cat(blocks, dim=0)[:num_features].contiguous()


class HetSNGPHead2d(nn.Module):
    """Per-pixel classifier replacing the decoder's 1x1 conv (`decoder.head`).

    head_type:
        "hetsngp"  GP output layer + heteroscedastic noise (the paper's model)
        "sngp"     GP output layer only
        "het"      heteroscedastic noise on a linear (1x1 conv) output layer
    forward(x) maps decoder features [B, F, H, W] to [B, C, H, W] log
    probabilities (or plain logits for "sngp" before the covariance is fitted).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        head_type: str = "hetsngp",
        num_inducing: int = 1024,
        kernel_scale: float = 1.0,
        normalize_input: bool = True,
        num_factors: int = 6,
        temperature: float = 1.0,
        train_mc_samples: int = 32,
        test_mc_samples: int = 256,
        mc_chunk: int = 64,
        rff_seed: int = 0,
    ):
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
        self.head_type = head_type
        self.use_gp = head_type in ("sngp", "hetsngp")
        self.use_het = head_type in ("het", "hetsngp")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes  # Conv2d-compatible for model/features.py
        self.num_factors = num_factors
        self.temperature = float(temperature)
        self.train_mc_samples = int(train_mc_samples)
        self.test_mc_samples = int(test_mc_samples)
        self.mc_chunk = int(mc_chunk)
        self.kernel_scale = float(kernel_scale)

        if self.use_gp:
            self.num_inducing = num_inducing
            self.input_norm = LayerNorm2d(in_channels) if normalize_input else nn.Identity()
            g = torch.Generator().manual_seed(rff_seed)
            self.register_buffer("rff_weight", orthogonal_random_features(num_inducing, in_channels, g))
            self.register_buffer("rff_bias", torch.rand(num_inducing, generator=g) * (2.0 * math.pi))
            # beta: the GP output layer, prior N(0, I) (Eq. 4). No bias -- f = Phi^T beta.
            self.weight = nn.Parameter(torch.empty(num_classes, num_inducing))
            nn.init.xavier_uniform_(self.weight)  # Keras Dense default, as in edward2
            self.bias = None
        else:
            self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
            self.bias = nn.Parameter(torch.zeros(num_classes))
            nn.init.xavier_uniform_(self.weight)

        if self.use_het:
            self.scale_layer = nn.Conv2d(in_channels, num_classes * num_factors, kernel_size=1)
            self.diag_layer = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            for layer in (self.scale_layer, self.diag_layer):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Laplace posterior covariance Sigma_c, [C, m, m] once fitted; empty until
        # then so unfitted checkpoints stay small (resized on load, see below).
        self.register_buffer("covariance", torch.zeros(0))

    # ---- state -------------------------------------------------------------
    @property
    def gp_fitted(self) -> bool:
        return self.use_gp and self.covariance.numel() > 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "covariance"
        if key in state_dict and state_dict[key].shape != self.covariance.shape:
            self.covariance = torch.empty_like(state_dict[key], device=self.covariance.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        s = f"{self.head_type}, in={self.in_channels}, classes={self.num_classes}"
        if self.use_gp:
            s += f", m={self.num_inducing}, kernel_scale={self.kernel_scale}, fitted={self.gp_fitted}"
        if self.use_het:
            s += (f", factors={self.num_factors}, tau={self.temperature}, "
                  f"S_train={self.train_mc_samples}, S_test={self.test_mc_samples}")
        return s

    # ---- building blocks -----------------------------------------------------
    def random_features(self, x: torch.Tensor) -> torch.Tensor:
        """Phi [B, m, H, W] of features x [B, F, H, W] (fp32)."""
        h = self.input_norm(x) / math.sqrt(self.kernel_scale)
        proj = torch.einsum("bfhw,mf->bmhw", h, self.rff_weight) + self.rff_bias[None, :, None, None]
        return math.sqrt(2.0 / self.num_inducing) * torch.cos(proj)

    def phi_and_loc(self, x: torch.Tensor):
        """(Phi or None, posterior-mode logits [B, C, H, W]) -- fp32."""
        if self.use_gp:
            phi = self.random_features(x)
            return phi, torch.einsum("bmhw,cm->bchw", phi, self.weight)
        return None, torch.einsum("bfhw,cf->bchw", x, self.weight) + self.bias[None, :, None, None]

    def gp_variance(self, phi: torch.Tensor) -> torch.Tensor:
        """Phi^T Sigma_c Phi per pixel and class, [B, C, H, W]."""
        b, m, h, w = phi.shape
        flat = phi.permute(0, 2, 3, 1).reshape(-1, m)
        var = torch.stack([((flat @ cov) * flat).sum(dim=1) for cov in self.covariance], dim=1)
        return var.clamp_min(0.0).reshape(b, h, w, -1).permute(0, 3, 1, 2)

    def _mc_log_mean_prob(self, loc, diag, factor, num_samples: int) -> torch.Tensor:
        """log(1/S sum_s softmax(u_s / tau)) with u_s = loc + diag*eps_K + factor eps_R."""
        b, c, h, w = loc.shape
        # One pass when training (S is small and autograd needs the whole graph);
        # chunks at eval, where S is large and memory would scale with it.
        chunk = num_samples if self.training else self.mc_chunk
        total = None
        for start in range(0, num_samples, chunk):
            s = min(chunk, num_samples - start)
            u = loc.unsqueeze(1)
            if diag is not None:
                u = u + diag.unsqueeze(1) * torch.randn(b, s, c, h, w, device=loc.device, dtype=loc.dtype)
            if factor is not None:
                eps_r = torch.randn(b, s, self.num_factors, h, w, device=loc.device, dtype=loc.dtype)
                u = u + torch.einsum("bcrhw,bsrhw->bschw", factor, eps_r)
            lse = torch.logsumexp(F.log_softmax(u / self.temperature, dim=2), dim=1)
            total = lse if total is None else torch.logaddexp(total, lse)
        return total - math.log(num_samples)

    # ---- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            phi, loc = self.phi_and_loc(x)
            gp_var = self.gp_variance(phi) if (self.gp_fitted and not self.training) else None
            if not self.use_het and gp_var is None:
                return loc  # SNGP at its posterior mode (training, or before fitting)

            b, c, h, w = loc.shape
            diag_var = None
            factor = None
            if self.use_het:
                diag = F.softplus(self.diag_layer(x)) + MIN_SCALE_MONTE_CARLO
                diag_var = diag.square()
                factor = self.scale_layer(x).view(b, c, self.num_factors, h, w)
            if gp_var is not None:
                diag_var = gp_var if diag_var is None else diag_var + gp_var
            diag = diag_var.sqrt() if diag_var is not None else None
            samples = self.train_mc_samples if self.training else self.test_mc_samples
            return self._mc_log_mean_prob(loc, diag, factor, samples)

    @torch.no_grad()
    def uncertainty_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-pixel uncertainty components at the decoder resolution, [B, H, W] each.

        gp_var   mean over classes of Phi^T Sigma_c Phi (model / distance uncertainty)
        het_var  mean over classes of the noise variance diag(V V^T + d^2) (data uncertainty)
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            out: dict[str, torch.Tensor] = {}
            phi, _ = self.phi_and_loc(x)
            if self.gp_fitted:
                out["gp_var"] = self.gp_variance(phi).mean(dim=1)
            if self.use_het:
                b, _, h, w = x.shape
                diag = F.softplus(self.diag_layer(x)) + MIN_SCALE_MONTE_CARLO
                factor = self.scale_layer(x).view(b, self.num_classes, self.num_factors, h, w)
                out["het_var"] = (diag.square() + factor.square().sum(dim=2)).mean(dim=1)
            return out


class _StopForward(Exception):
    pass


@torch.no_grad()
def fit_laplace_covariance(
    model: nn.Module,
    head: HetSNGPHead2d,
    loader,
    *,
    device,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
    ignore_index: int = 255,
    max_batches: Optional[int] = None,
    log_every: int = 200,
) -> dict:
    """Fit Sigma_c = (I + sum_i w_i p_ic (1 - p_ic) Phi_i Phi_i^T)^-1 (Eq. 5) in place.

    Data points are the decoder-resolution pixels of `loader` (the training set,
    without augmentation). The CE loss is evaluated on the bilinearly upsampled
    logits, so one decoder pixel stands for the full-resolution pixels it
    covers: w_i is the number of labelled (non-ignore) full-resolution pixels
    in its cell. p is the softmax of the posterior-mode logits Phi^T beta_hat /
    tau (Eq. 5 with tau = 1); the 1/tau^2 factor makes the Hessian exact for a
    tempered softmax. Accumulated in float64 and inverted by Cholesky.
    """
    try:
        from vision_backend.training.utils import parse_segmentation_batch
    except ModuleNotFoundError:
        from training.utils import parse_segmentation_batch

    if not head.use_gp:
        raise ValueError("fit_laplace_covariance needs a GP head ('sngp' or 'hetsngp').")
    model.eval()
    m, c = head.num_inducing, head.num_classes
    precision = torch.zeros(c, m, m, dtype=torch.float64, device=device)
    captured: dict[str, torch.Tensor] = {}

    def _grab(module, args):
        captured["x"] = args[0]
        raise _StopForward

    handle = head.register_forward_pre_hook(_grab)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    n_pixels, n_batches, t0 = 0.0, 0, time.time()
    try:
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            n_batches += 1
            local, target, context = parse_segmentation_batch(batch)
            local = local.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True)
            args = (local,) if context is None else (local, context.to(device, non_blocking=True).float())
            try:
                with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    model(*args)
            except _StopForward:
                pass
            x = captured.pop("x").float()
            phi, loc = head.phi_and_loc(x)
            # labelled full-resolution pixels per decoder cell
            valid = (target != ignore_index).float().unsqueeze(1)
            weight = F.adaptive_avg_pool2d(valid, phi.shape[-2:]) * (
                valid.shape[-2] * valid.shape[-1] / (phi.shape[-2] * phi.shape[-1]))
            p = torch.softmax(loc / head.temperature, dim=1)
            hess = p * (1.0 - p) / head.temperature ** 2 * weight  # [B, C, h, w]
            flat_phi = phi.permute(0, 2, 3, 1).reshape(-1, m)
            flat_h = hess.permute(0, 2, 3, 1).reshape(-1, c)
            for k in range(c):
                precision[k] += (flat_phi * flat_h[:, k:k + 1]).T.matmul(flat_phi).double()
            n_pixels += float(weight.sum())
            if log_every and n_batches % log_every == 0:
                print(f"[laplace] {n_batches} batches, {n_pixels:.3g} labelled pixels, {time.time() - t0:.0f}s",
                      flush=True)
    finally:
        handle.remove()
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    eye = torch.eye(m, dtype=torch.float64, device=device)
    cov = torch.cholesky_inverse(torch.linalg.cholesky(precision + eye))
    head.covariance = cov.float().to(head.rff_weight.device)
    return {
        "labelled_pixels": n_pixels,
        "batches": n_batches,
        "seconds": time.time() - t0,
        # mean_c trace(Sigma_c) / m: 1.0 = prior only, -> 0 as data pins beta down
        "mean_relative_posterior_variance": float(cov.diagonal(dim1=1, dim2=2).mean()),
    }
