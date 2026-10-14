"""Diagonal Laplace prior: the previous campaign's posterior as the next prior.

Sequential Bayesian updating across labelling campaigns (online Laplace:
Ritter, Botev & Barber, NeurIPS 2018; as Gaussian filtering in weight space:
Sliwa, Schneider, Bosch, Kristiadi & Hennig, "Low-Rank Filtering and Smoothing
for Sequential Deep Learning", arXiv:2410.06800). After campaign t the weights
have a Gaussian posterior N(mu_t, Lambda_t^-1); campaign t+1 minimises

    sum_n CE_n(theta) + 1/2 (theta - mu_t)^T Lambda_t (theta - mu_t)

so moving a weight costs in proportion to how well the previous data pinned it
down. This is the diagonal version (EWC done as a proper online Laplace); the
paper's diagonal-plus-low-rank precision is the upgrade path.

Campaign 0 is the NOAH-H segmentation. Its precision is the Fisher -- equal
to the generalised Gauss-Newton for a softmax likelihood -- estimated per
image with labels SAMPLED from the model (an unbiased Monte Carlo estimate;
the "empirical Fisher" with the given labels is not, Kunstner et al. 2019):

    F_img = E_x[ (grad_theta sum_pixels log p(y_i | x)) ** 2 ],  y_i ~ p(. | x)

The pixels of an image are far from independent, so the full-data precision
(55 k images x 262 k pixels) would freeze the network onto NOAH-H's own
errors. Instead Lambda = tau * F_img with tau = the effective number of NOAH-H
images, a tempering (Wenzel et al. 2020; Kirkpatrick et al.'s lambda) that is
chosen on the expert val split. scope="encoder" leaves the decoder and the
segmentation head without a prior -- the extreme of the paper's layer-wise
transition noise -- because E13 showed the features transfer to new sites
while NOAH-H's class mapping does not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def mc_fisher_diagonal_segmentation(
    seg_model: torch.nn.Module,
    batches: Iterable[dict],
    *,
    device: torch.device,
    ignore_index: int = 255,
    amp: bool = True,
    log_every: int = 200,
) -> tuple[dict[str, torch.Tensor], int]:
    """Per-image diagonal MC Fisher of a segmentation model, averaged over images.

    `batches` must yield ONE image each ({"image": [1,1,H,W], "label": [1,H,W]}):
    the diagonal needs per-example gradients, and batch size 1 is the exact way
    to get them for any architecture.
    """
    seg_model.eval()
    params = {n: p for n, p in seg_model.named_parameters() if p.requires_grad}
    acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    count = 0
    for batch in batches:
        image = batch["image"].to(device)
        if image.shape[0] != 1:
            raise ValueError("mc_fisher_diagonal_segmentation needs batches of one image")
        valid = batch["label"].to(device) != ignore_index
        if not valid.any():
            continue
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = seg_model(image)
        logp = F.log_softmax(logits.float(), dim=1)                       # [1, C, H, W]
        with torch.no_grad():
            y = torch.distributions.Categorical(logits=logp.permute(0, 2, 3, 1)).sample()  # [1, H, W]
        loglik = logp.gather(1, y[:, None]).squeeze(1)[valid].sum()
        seg_model.zero_grad(set_to_none=True)
        loglik.backward()
        for n, p in params.items():
            if p.grad is not None:
                acc[n] += p.grad.detach().float() ** 2
        count += 1
        if log_every and count % log_every == 0:
            print(f"[laplace] Fisher: {count} images")
    seg_model.zero_grad(set_to_none=True)
    if count == 0:
        raise ValueError("no labelled pixels in any image")
    return {n: a / count for n, a in acc.items()}, count


def mc_fisher_diagonal_blocks(
    model: torch.nn.Module,
    batches: Iterable[dict],
    *,
    device: torch.device,
    amp: bool = True,
    log_every: int = 200,
) -> tuple[dict[str, torch.Tensor], int]:
    """Per-block diagonal MC Fisher of a block classifier (all trainable
    parameters), averaged over blocks; the label is sampled from the model's
    own block predictive. `batches` must yield ONE block each ({"image",
    "valid"}), as in `mc_fisher_diagonal_segmentation`. Each block is one
    observation, so the data precision of N blocks is N times the average."""
    model.eval()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    count = 0
    for batch in batches:
        image = batch["image"].to(device)
        if image.shape[0] != 1:
            raise ValueError("mc_fisher_diagonal_blocks needs batches of one block")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(image, batch["valid"].to(device))
        logp = F.log_softmax(logits.float(), dim=1)                       # [1, K]
        with torch.no_grad():
            y = torch.distributions.Categorical(logits=logp).sample()     # [1]
        model.zero_grad(set_to_none=True)
        logp[0, y[0]].backward()
        for n, p in params.items():
            if p.grad is not None:
                acc[n] += p.grad.detach().float() ** 2
        count += 1
        if log_every and count % log_every == 0:
            print(f"[laplace] Fisher: {count} blocks")
    model.zero_grad(set_to_none=True)
    if count == 0:
        raise ValueError("no blocks")
    return {n: a / count for n, a in acc.items()}, count


def save_prior(path: str | Path, mean: dict[str, torch.Tensor], fisher: dict[str, torch.Tensor], **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": {n: t.detach().float().cpu() for n, t in mean.items()},
                "fisher": {n: t.detach().float().cpu() for n, t in fisher.items()}, **meta}, str(path))


def quadratic_form(params: dict[str, torch.Tensor], prior: dict, *, scope: str = "all") -> float:
    """1/2 sum_j F_j (theta_j - mu_j)^2 over the prior's parameters (tau = 1)."""
    total = 0.0
    for n, theta in params.items():
        if n in prior["fisher"] and in_scope(n, scope):
            total += 0.5 * float((prior["fisher"][n] * (theta.float().cpu() - prior["mean"][n]) ** 2).sum())
    return total


def in_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "encoder":  # segmenter names, bare or inside a BlockClassifier ("seg.encoder.")
        return name.removeprefix("seg.").startswith("encoder.")
    raise ValueError(f"unknown Laplace scope {scope!r} (all | encoder)")


def covers_block_head(prior: dict) -> bool:
    """True for a posterior over a whole BlockClassifier ('seg.*' and head names,
    scripts/fit_block_artifacts.py), False for a prior over a bare segmenter."""
    return any(n.startswith("seg.") for n in prior["fisher"])


class DiagonalLaplacePrior:
    """1/2 sum_j (tau * F_j + delta) (theta_j - mu_j)^2 over a model's parameters."""

    def __init__(self, path: str | Path | dict, model: torch.nn.Module, *, tau: float, delta: float = 0.0,
                 scope: str = "all", device: Optional[torch.device] = None, atol: float = 1e-5):
        prior = path if isinstance(path, dict) else torch.load(str(path), map_location="cpu", weights_only=False)
        self.terms: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
        self.num_params = 0
        max_offset = 0.0
        for n, p in model.named_parameters():
            if not in_scope(n, scope):
                continue
            if n not in prior["fisher"]:
                raise KeyError(f"prior {path} has no entry for parameter {n!r}")
            mu = prior["mean"][n]
            max_offset = max(max_offset, float((p.detach().float().cpu() - mu).abs().max()))
            lam = tau * prior["fisher"][n] + delta
            if float(lam.max()) <= 0:
                continue
            dev = device or p.device
            self.terms.append((p, mu.to(dev), lam.to(dev)))
            self.num_params += p.numel()
        if max_offset > atol:
            raise ValueError(f"the model is {max_offset:.2e} away from the prior mean -- it must start from the "
                             f"segmenter the prior was fitted on ({prior.get('source_checkpoint')})")
        self.tau, self.delta, self.scope = tau, delta, scope

    def penalty(self) -> torch.Tensor:
        total = None
        for p, mu, lam in self.terms:
            term = (lam * (p.float() - mu) ** 2).sum()
            total = term if total is None else total + term
        return 0.5 * total if total is not None else torch.zeros(())
