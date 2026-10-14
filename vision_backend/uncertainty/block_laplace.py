"""Last-layer Laplace approximation for the Stage-4 block classifier.

The block logits are  f(x) = offset(x) + W phi~(x),  phi~ = [phi, 1],
W = [fc.weight | fc.bias]  (K x (D+1)), with phi and offset from
`BlockClassifier.block_features`. Keeping everything below `fc` at the MAP,
a Gaussian posterior over W -- the last-layer Laplace that Laplace Redux
(Daxberger et al., NeurIPS 2021, arXiv:2106.14806) recommends as the default --
gives every new block a Gaussian over its logits: mean f(x), covariance
J Sigma J^T with J = I_K (x) phi~^T.

Posterior precision = GGN + lambda I with GGN = sum_n kron(Lambda_n, phi~_n phi~_n^T),
Lambda_n = diag(p_n) - p_n p_n^T, summed over the training blocks. It is exact
(not KFAC): K (D+1) = 18 x 271 = 4 878 parameters. The GGN is stored as its
eigendecomposition, so the prior precision lambda can be re-tuned without
refitting: Sigma(lambda) = V diag(1 / (e + lambda)) V^T.

Softmax ignores a shift shared by all logits. Those directions are in the
GGN's null space, so their posterior variance is the prior's 1/lambda, which
would only inflate the probit approximation. The logit covariance is
therefore centred (C Sigma_f C, C = I - 11^T/K): the Monte Carlo predictive
is unchanged, the spurious term leaves the probit.

Predictive: the probit approximation (MacKay 1992; softmax of
mu_k / sqrt(1 + pi/8 var_k)), or Monte Carlo over the logit Gaussian, which
also splits the predictive entropy into expected entropy (aleatoric) and
mutual information (epistemic).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

SCHEMA_VERSION = 1
_CHUNK = 64


def _with_bias(phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)


def last_layer_ggn(phi: torch.Tensor, logits: torch.Tensor, *, chunk: int = 256) -> torch.Tensor:
    """sum_n kron(diag(p_n) - p_n p_n^T, phi~_n phi~_n^T), float64, [K(D+1), K(D+1)].

    Parameter order is W = [fc.weight | fc.bias] flattened row-major
    (index k * (D+1) + d). Computed as one GEMM per chunk:
    [K^2, n] @ [n, (D+1)^2], then permuted to [K, D+1, K, D+1].
    """
    phi, logits = phi.double(), logits.double()
    n, d = phi.shape
    k = logits.shape[1]
    out = torch.zeros(k * k, (d + 1) ** 2, dtype=torch.float64, device=phi.device)
    for s in range(0, n, chunk):
        p = torch.softmax(logits[s:s + chunk], dim=1)
        lam = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]           # [n, K, K]
        x = _with_bias(phi[s:s + chunk])
        outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)         # [n, (D+1)^2]
        out += lam.reshape(len(x), -1).T @ outer
    d1 = d + 1
    return out.reshape(k, k, d1, d1).permute(0, 2, 1, 3).reshape(k * d1, k * d1)


@dataclass
class LastLayerLaplace:
    """Gaussian posterior over the block head's last layer; see the module docstring."""

    eigvecs: torch.Tensor          # [P, P]; column p is an eigenvector of the GGN (row = parameter index)
    eigvals: torch.Tensor          # [P]
    prior_precision: float
    num_classes: int
    feature_dim: int               # D, without the bias input
    n_fit: int
    weight: torch.Tensor           # [K, D+1], the MAP [fc.weight | fc.bias] the GGN was fitted at
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, phi: torch.Tensor, logits: torch.Tensor, weight: torch.Tensor, *,
            prior_precision: float = 1.0, meta: Optional[dict] = None) -> "LastLayerLaplace":
        ggn = last_layer_ggn(phi, logits)
        ggn = 0.5 * (ggn + ggn.T)
        e, v = torch.linalg.eigh(ggn)
        return cls(eigvecs=v.float(), eigvals=e.clamp_min(0).float(), prior_precision=float(prior_precision),
                   num_classes=int(logits.shape[1]), feature_dim=int(phi.shape[1]), n_fit=int(phi.shape[0]),
                   weight=weight.detach().float().cpu(), meta=dict(meta or {}))

    def to(self, device) -> "LastLayerLaplace":
        return LastLayerLaplace(self.eigvecs.to(device), self.eigvals.to(device), self.prior_precision,
                                self.num_classes, self.feature_dim, self.n_fit, self.weight, self.meta)

    def check_model(self, fc: torch.nn.Linear, atol: float = 1e-5) -> None:
        """Raise unless `fc` is the layer this posterior was fitted at."""
        w = torch.cat([fc.weight.detach(), fc.bias.detach()[:, None]], dim=1).float().cpu()
        if w.shape != self.weight.shape or float((w - self.weight).abs().max()) > atol:
            raise ValueError("this last-layer Laplace was fitted for a different block head")

    def logit_covariance(self, phi: torch.Tensor, *, prior_precision: Optional[float] = None,
                         centered: bool = True) -> torch.Tensor:
        """Posterior covariance of the logits of each block, [B, K, K]."""
        lam = self.prior_precision if prior_precision is None else float(prior_precision)
        k, d1 = self.num_classes, self.feature_dim + 1
        if phi.shape[1] != self.feature_dim:
            raise ValueError(f"phi must be [B, {self.feature_dim}], got {tuple(phi.shape)}")
        v = self.eigvecs.to(phi.device).view(k, d1, -1)
        s = 1.0 / (self.eigvals.to(phi.device) + lam)
        out = []
        for chunk in phi.float().split(_CHUNK):
            jv = torch.einsum("kdp,bd->bkp", v, _with_bias(chunk))           # [b, K, P]
            out.append(torch.einsum("bkp,p,blp->bkl", jv, s, jv))
        cov = torch.cat(out) if out else phi.new_zeros(0, k, k)
        if centered:
            c = torch.eye(k, device=phi.device) - 1.0 / k
            cov = c @ cov @ c
        return 0.5 * (cov + cov.transpose(1, 2))

    def predictive(self, logits: torch.Tensor, phi: torch.Tensor, *, method: str = "probit",
                   n_samples: int = 1000, prior_precision: Optional[float] = None,
                   generator: Optional[torch.Generator] = None) -> dict:
        """Posterior predictive for blocks with MAP logits [B, K] and features phi [B, D].

        method "probit": {"probs", "entropy", "logit_var"}.
        method "mc": additionally "aleatoric" (expected entropy) and
        "mutual_information" (epistemic: entropy - aleatoric).
        """
        cov = self.logit_covariance(phi, prior_precision=prior_precision)
        logits = logits.float()
        var = cov.diagonal(dim1=1, dim2=2)
        if method == "probit":
            probs = torch.softmax(logits / torch.sqrt(1.0 + math.pi / 8.0 * var), dim=1)
            return {"probs": probs, "entropy": _entropy(probs), "logit_var": var}
        if method != "mc":
            raise ValueError(f"unknown method {method!r} (probit | mc)")
        ev, q = torch.linalg.eigh(cov.double())
        scale = q * ev.clamp_min(0).sqrt()[:, None, :]                       # [B, K, K]
        eps = torch.randn(logits.shape[0], n_samples, logits.shape[1], generator=generator,
                          dtype=torch.float64, device="cpu").to(logits.device)
        samples = logits.double()[:, None, :] + eps @ scale.transpose(1, 2)  # [B, S, K]
        p = torch.softmax(samples, dim=2)
        probs = p.mean(1)
        aleatoric = _entropy(p).mean(1)
        entropy = _entropy(probs)
        return {"probs": probs.float(), "entropy": entropy.float(), "logit_var": var,
                "aleatoric": aleatoric.float(), "mutual_information": (entropy - aleatoric).clamp_min(0).float()}


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


def tune_prior_precision(la: LastLayerLaplace, logits: torch.Tensor, phi: torch.Tensor, labels: torch.Tensor,
                         grid: Optional[torch.Tensor] = None) -> tuple[float, list[tuple[float, float]]]:
    """lambda minimising the probit predictive NLL on held-out blocks; returns (lambda*, [(lambda, nll)])."""
    grid = torch.logspace(-4, 6, 41) if grid is None else grid
    table = []
    for lam in grid.tolist():
        probs = la.predictive(logits, phi, method="probit", prior_precision=lam)["probs"]
        nll = float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels.to(probs.device)))
        table.append((lam, nll))
    best = min(table, key=lambda t: t[1])[0]
    return best, table


def save_last_layer_laplace(la: LastLayerLaplace, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "last_layer_laplace_ggn", "schema_version": SCHEMA_VERSION,
                "eigvecs": la.eigvecs.cpu(), "eigvals": la.eigvals.cpu(), "prior_precision": la.prior_precision,
                "num_classes": la.num_classes, "feature_dim": la.feature_dim, "n_fit": la.n_fit,
                "weight": la.weight.cpu(), "meta": la.meta}, str(path))


def load_last_layer_laplace(path: str | Path) -> LastLayerLaplace:
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if raw.get("kind") != "last_layer_laplace_ggn":
        raise ValueError(f"{path}: not a last-layer Laplace file ({raw.get('kind')!r})")
    return LastLayerLaplace(raw["eigvecs"], raw["eigvals"], float(raw["prior_precision"]), int(raw["num_classes"]),
                            int(raw["feature_dim"]), int(raw["n_fit"]), raw["weight"], raw.get("meta", {}))
