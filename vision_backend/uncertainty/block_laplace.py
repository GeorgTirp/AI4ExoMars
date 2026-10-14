"""Last-layer Laplace approximation for the Stage-4 block classifier.

The block logits are  f(x) = offset(x) + W phi~(x),  phi~ = [phi, 1],
W = [fc.weight | fc.bias]  (K x (D+1)), with phi and offset from
`BlockClassifier.block_features`. Keeping everything below `fc` at the MAP,
a Gaussian posterior over W -- the last-layer Laplace that Laplace Redux
(Daxberger et al., NeurIPS 2021, arXiv:2106.14806) recommends as the default --
gives every new block a Gaussian over its logits: mean f(x), covariance
J Sigma J^T with J = I_K (x) phi~^T.

Posterior precision = GGN + lambda I with GGN = sum_n kron(Lambda_n, phi~_n phi~_n^T)
over the training blocks; Lambda_n is the Fisher of the likelihood in logit
space:
- link "softmax" (categorical): Lambda_n = diag(p_n) - p_n p_n^T;
- link "normcdf" (one Bernoulli per class, p = Phi(f), Mucsanyi, Da Costa &
  Hennig, NeurIPS 2025): Lambda_n = diag(phi(f)^2 / (Phi(f) Phi(-f))) -- the
  classes decouple.
It is exact (not KFAC): K (D+1) = 18 x 271 = 4 878 parameters, stored as the
GGN's eigendecomposition so the prior precision lambda can be re-tuned without
refitting: Sigma(lambda) = V diag(1 / (e + lambda)) V^T.

Softmax ignores a shift shared by all logits; those directions are in the GGN's
null space with posterior variance 1/lambda, which would only inflate the
probit approximation. For softmax the logit covariance is therefore centred
(C Sigma_f C, C = I - 11^T/K); the Monte Carlo predictive is unchanged.
normalize(Phi(f)) has no such invariance, so normcdf is not centred.

Predictives (`predictive(method=...)`):
- "probit": softmax -- the multi-class probit heuristic softmax(mu / sqrt(1 + pi/8 var))
  (exact only for two classes); normcdf -- E[Phi(f_k)] = Phi(mu_k / sqrt(1 + var_k)),
  exact per class, then normalised: a ratio of expectations, which Mucsanyi et al.
  bound by the variance of sum_c Phi(f_c) (accurate unless the posterior is very broad).
- "mc": Monte Carlo over the logit Gaussian (unbiased for both links); also
  the expected entropy (aleatoric) and mutual information (epistemic).
- "dirichlet" (normcdf): the moment-matched Dirichlet of Mucsanyi et al. --
  closed-form E[Phi(f)^2] = Phi(h) - 2 T(h, 1/sqrt(1 + 2 var)) (Owen's T),
  h = mu / sqrt(1 + var); Minka's moment estimate of the Dirichlet precision,
  geometric mean over classes. Deterministic expected entropy and mutual
  information, no sampling.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

SCHEMA_VERSION = 2
_CHUNK = 64
LINKS = ("softmax", "normcdf")


def _with_bias(phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)


def _logit_fisher(logits: torch.Tensor, link: str) -> torch.Tensor:
    """Fisher of the likelihood w.r.t. the logits, [n, K, K]."""
    if link == "softmax":
        p = torch.softmax(logits, dim=1)
        return torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    if link == "normcdf":
        log_pdf = -0.5 * logits ** 2 - 0.5 * math.log(2 * math.pi)
        lam = torch.exp(2 * log_pdf - torch.special.log_ndtr(logits) - torch.special.log_ndtr(-logits))
        return torch.diag_embed(lam)
    raise ValueError(f"unknown link {link!r}")


def last_layer_ggn(phi: torch.Tensor, logits: torch.Tensor, *, link: str = "softmax",
                   chunk: int = 256) -> torch.Tensor:
    """sum_n kron(Lambda_n, phi~_n phi~_n^T), float64, [K(D+1), K(D+1)].

    Parameter order is W = [fc.weight | fc.bias] flattened row-major
    (index k * (D+1) + d). Computed as one GEMM per chunk:
    [K^2, n] @ [n, (D+1)^2], then permuted to [K, D+1, K, D+1].
    """
    phi, logits = phi.double(), logits.double()
    n, d = phi.shape
    k = logits.shape[1]
    out = torch.zeros(k * k, (d + 1) ** 2, dtype=torch.float64, device=phi.device)
    for s in range(0, n, chunk):
        lam = _logit_fisher(logits[s:s + chunk], link)                       # [n, K, K]
        x = _with_bias(phi[s:s + chunk])
        outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)         # [n, (D+1)^2]
        out += lam.reshape(len(x), -1).T @ outer
    d1 = d + 1
    return out.reshape(k, k, d1, d1).permute(0, 2, 1, 3).reshape(k * d1, k * d1)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


def normcdf_moments(mu: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E[Phi(f)] and E[Phi(f)^2] for f ~ N(mu, var), elementwise (float64)."""
    from scipy.special import owens_t

    mu, var = mu.double(), var.double().clamp_min(0)
    h = mu / torch.sqrt(1 + var)
    m1 = torch.special.ndtr(h)
    a = 1.0 / torch.sqrt(1 + 2 * var)
    t = torch.from_numpy(owens_t(h.cpu().numpy(), a.cpu().numpy())).to(mu.device)
    return m1, (m1 - 2 * t).clamp_min(0)


def dirichlet_from_normcdf(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Moment-matched Dirichlet concentration gamma [B, K] of normalize(Phi(f)),
    f_k ~ N(mu_k, var_k) independent (Mucsanyi, Da Costa & Hennig 2025)."""
    m1, m2 = normcdf_moments(mu, var)
    s = m1.sum(1, keepdim=True)
    scale = s.clamp_min(1.0)
    per_class = (m1 * scale - m2).clamp_min(1e-12) / (m2 - m1 ** 2).clamp_min(1e-12)
    gamma0 = torch.exp(torch.log(per_class).mean(1, keepdim=True))         # geometric mean over classes
    return gamma0 * m1 / s


def dirichlet_uncertainty(gamma: torch.Tensor) -> dict:
    """Mean, entropy of the mean, expected entropy and mutual information of Dir(gamma)."""
    g0 = gamma.sum(1, keepdim=True)
    m = gamma / g0
    expected = -(m * (torch.digamma(gamma + 1) - torch.digamma(g0 + 1))).sum(1)
    entropy = _entropy(m)
    return {"probs": m, "entropy": entropy, "aleatoric": expected,
            "mutual_information": (entropy - expected).clamp_min(0)}


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
    link: str = "softmax"
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, phi: torch.Tensor, logits: torch.Tensor, weight: torch.Tensor, *, link: str = "softmax",
            prior_precision: float = 1.0, meta: Optional[dict] = None) -> "LastLayerLaplace":
        if link not in LINKS:
            raise ValueError(f"unknown link {link!r}")
        ggn = last_layer_ggn(phi, logits, link=link)
        ggn = 0.5 * (ggn + ggn.T)
        e, v = torch.linalg.eigh(ggn)
        return cls(eigvecs=v.float(), eigvals=e.clamp_min(0).float(), prior_precision=float(prior_precision),
                   num_classes=int(logits.shape[1]), feature_dim=int(phi.shape[1]), n_fit=int(phi.shape[0]),
                   weight=weight.detach().float().cpu(), link=link, meta=dict(meta or {}))

    def to(self, device) -> "LastLayerLaplace":
        return dataclasses.replace(self, eigvecs=self.eigvecs.to(device), eigvals=self.eigvals.to(device))

    def check_model(self, fc: torch.nn.Linear, atol: float = 1e-5) -> None:
        """Raise unless `fc` is the layer this posterior was fitted at."""
        w = torch.cat([fc.weight.detach(), fc.bias.detach()[:, None]], dim=1).float().cpu()
        if w.shape != self.weight.shape or float((w - self.weight).abs().max()) > atol:
            raise ValueError("this last-layer Laplace was fitted for a different block head")

    def jacobian_basis(self, phi: torch.Tensor) -> torch.Tensor:
        """J V for each block, [B, K, P] -- independent of lambda, so it can be reused across a lambda sweep."""
        if phi.shape[1] != self.feature_dim:
            raise ValueError(f"phi must be [B, {self.feature_dim}], got {tuple(phi.shape)}")
        v = self.eigvecs.to(phi.device).view(self.num_classes, self.feature_dim + 1, -1)
        return torch.cat([torch.einsum("kdp,bd->bkp", v, _with_bias(c)) for c in phi.float().split(_CHUNK)])

    def logit_covariance(self, phi: Optional[torch.Tensor] = None, *, prior_precision: Optional[float] = None,
                         jv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Posterior covariance of the logits of each block, [B, K, K] (centred for softmax)."""
        lam = self.prior_precision if prior_precision is None else float(prior_precision)
        jv = self.jacobian_basis(phi) if jv is None else jv
        s = 1.0 / (self.eigvals.to(jv.device) + lam)
        cov = torch.cat([torch.einsum("bkp,p,blp->bkl", c, s, c) for c in jv.split(_CHUNK)]) if len(jv) \
            else jv.new_zeros(0, self.num_classes, self.num_classes)
        if self.link == "softmax":
            c = torch.eye(self.num_classes, device=jv.device) - 1.0 / self.num_classes
            cov = c @ cov @ c
        return 0.5 * (cov + cov.transpose(1, 2))

    def predictive(self, logits: torch.Tensor, phi: Optional[torch.Tensor] = None, *, method: str = "probit",
                   n_samples: int = 1000, prior_precision: Optional[float] = None,
                   generator: Optional[torch.Generator] = None, jv: Optional[torch.Tensor] = None) -> dict:
        """Posterior predictive for blocks with MAP logits [B, K] and features phi [B, D] (or a cached `jv`).

        method "probit": {"probs", "entropy", "logit_var"} (closed form; exact per class for normcdf).
        method "mc": additionally "aleatoric" (expected entropy) and "mutual_information".
        method "dirichlet" (normcdf only): the same keys, in closed form.
        """
        cov = self.logit_covariance(phi, prior_precision=prior_precision, jv=jv)
        return predictive_from_covariance(logits, cov, self.link, method=method, n_samples=n_samples,
                                          generator=generator)


def predictive_from_covariance(logits: torch.Tensor, cov: torch.Tensor, link: str, *, method: str = "probit",
                               n_samples: int = 1000, generator: Optional[torch.Generator] = None) -> dict:
    logits = logits.float().to(cov.device)
    var = cov.diagonal(dim1=1, dim2=2)
    if method == "probit":
        if link == "softmax":
            probs = torch.softmax(logits / torch.sqrt(1.0 + math.pi / 8.0 * var), dim=1)
        else:  # normalize(Phi(h)) computed as softmax(log Phi(h)): stable when every Phi underflows
            probs = torch.softmax(torch.special.log_ndtr(logits / torch.sqrt(1.0 + var)), dim=1)
        return {"probs": probs, "entropy": _entropy(probs), "logit_var": var}
    if method == "dirichlet":
        if link != "normcdf":
            raise ValueError("the closed-form Dirichlet is for the normcdf link")
        out = {k: v.float() for k, v in dirichlet_uncertainty(dirichlet_from_normcdf(logits, var)).items()}
        return out | {"logit_var": var}
    if method != "mc":
        raise ValueError(f"unknown method {method!r} (probit | mc | dirichlet)")
    ev, q = torch.linalg.eigh(cov.double())
    scale = q * ev.clamp_min(0).sqrt()[:, None, :]                       # [B, K, K]
    eps = torch.randn(logits.shape[0], n_samples, logits.shape[1], generator=generator,
                      dtype=torch.float64, device="cpu").to(logits.device)
    samples = logits.double()[:, None, :] + eps @ scale.transpose(1, 2)  # [B, S, K]
    if link == "softmax":
        p = torch.softmax(samples, dim=2)
    else:
        p = torch.softmax(torch.special.log_ndtr(samples), dim=2)
    probs = p.mean(1)
    aleatoric = _entropy(p).mean(1)
    entropy = _entropy(probs)
    return {"probs": probs.float(), "entropy": entropy.float(), "logit_var": var,
            "aleatoric": aleatoric.float(), "mutual_information": (entropy - aleatoric).clamp_min(0).float()}


def link_log_probs(logits: torch.Tensor, link: str = "softmax") -> torch.Tensor:
    """Class log-probabilities from logits: log-softmax, or log normalize(Phi(f)) for normcdf."""
    logits = logits.double()
    if link == "softmax":
        return torch.log_softmax(logits, dim=1)
    log_q = torch.special.log_ndtr(logits)
    return log_q - torch.logsumexp(log_q, dim=1, keepdim=True)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, link: str = "softmax") -> float:
    """argmin_T NLL of p(logits / T) on a log-spaced grid, refined once (Guo et al. 2017)."""
    logits, labels = torch.as_tensor(logits).double(), torch.as_tensor(labels).long()

    def nll(t):
        return float(-link_log_probs(logits / t, link)[torch.arange(len(labels)), labels].mean())

    grid = torch.exp(torch.linspace(math.log(0.05), math.log(20.0), 400)).tolist()
    t0 = min(grid, key=nll)
    fine = torch.exp(torch.linspace(math.log(t0 / 1.05), math.log(t0 * 1.05), 200)).tolist()
    return float(min(fine, key=nll))


def tune_prior_precision(la: LastLayerLaplace, logits: torch.Tensor, phi: torch.Tensor, labels: torch.Tensor,
                         grid: Optional[torch.Tensor] = None, *, method: str = "probit", n_samples: int = 1000,
                         seed: int = 0) -> tuple[float, list[tuple[float, float]]]:
    """lambda minimising the held-out NLL of the given predictive; returns (lambda*, [(lambda, nll)]).

    The Monte Carlo predictive uses the same random numbers for every lambda
    (common random numbers), so the sweep compares lambdas, not sampling noise.
    """
    grid = torch.logspace(-4, 6, 41) if grid is None else grid
    jv = la.jacobian_basis(phi)
    table = []
    for lam in grid.tolist():
        g = torch.Generator().manual_seed(seed)
        probs = la.predictive(logits, method=method, prior_precision=lam, jv=jv, n_samples=n_samples,
                              generator=g)["probs"]
        nll = float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels.to(probs.device)))
        table.append((lam, nll))
    best = min(table, key=lambda t: t[1])[0]
    return best, table


def save_last_layer_laplace(la: LastLayerLaplace, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "last_layer_laplace_ggn", "schema_version": SCHEMA_VERSION,
                "eigvecs": la.eigvecs.cpu(), "eigvals": la.eigvals.cpu(), "prior_precision": la.prior_precision,
                "num_classes": la.num_classes, "feature_dim": la.feature_dim, "n_fit": la.n_fit,
                "weight": la.weight.cpu(), "link": la.link, "meta": la.meta}, str(path))


def load_last_layer_laplace(path: str | Path) -> LastLayerLaplace:
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if raw.get("kind") != "last_layer_laplace_ggn":
        raise ValueError(f"{path}: not a last-layer Laplace file ({raw.get('kind')!r})")
    return LastLayerLaplace(raw["eigvecs"], raw["eigvals"], float(raw["prior_precision"]), int(raw["num_classes"]),
                            int(raw["feature_dim"]), int(raw["n_fit"]), raw["weight"],
                            link=raw.get("link", "softmax"), meta=raw.get("meta", {}))
