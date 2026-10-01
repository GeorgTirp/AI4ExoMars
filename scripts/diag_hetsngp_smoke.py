#!/usr/bin/env python3
"""GPU smoke test of the HetSNGP head before committing a 30-epoch run to it.

Trains the SimMIM HybridEncoder segmentation model exactly as the A/B arms ran
it (bf16 autocast, torch.compile(dynamic=False), NAdamW, batch 4, real cached
crops) for STEPS steps, once with the plain 1x1 classifier and once with the
HetSNGP head, and reports per configuration:

  - that compile + forward/backward run and the loss stays finite,
  - median step time and peak GPU memory (the head's real cost),
  - that eval-mode (MC) predictions are normalized log-probabilities,

then times the Laplace fit on a few batches (extrapolated to the full train
split) and checks the fitted predictive. Nothing is saved.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from vision_backend.model.hetsngp import fit_laplace_covariance
from vision_backend.model.optimizers import create_optimizer
from vision_backend.seg_dataset import create_segmentation_dataloaders
from vision_backend.training.builders import build_simmim_segmentation_model

DER = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived"
STEPS, WARMUP = 150, 30
BASE = {"model_kind": "simmim", "in_channels": 1, "global_base_grid": 32, "window_size": 8,
        "drop_path": 0.0, "decoder_channels": 256, "decoder_dropout": 0.1, "num_classes": 14}
HEAD = {"head_type": "hetsngp", "num_inducing": 1024, "kernel_scale": 1.0, "num_factors": 6,
        "temperature": 1.0, "train_mc_samples": 32, "test_mc_samples": 256}


def loader(augment: bool, batch_size: int = 4):
    kw = json.loads(Path(f"{DER}/seg_loader_DC_full.json").read_text())
    kw.update(batch_size=batch_size, num_workers=6, seed=42, augment=augment,
              persistent_workers=False, cache_dir=f"{DER}/seg_crop_cache_full")
    return create_segmentation_dataloaders(**kw)


def run(name, cfg, train_iter, device):
    torch._dynamo.reset()
    torch.manual_seed(42)
    base = build_simmim_segmentation_model(cfg).to(device).train()
    model = torch.compile(base, dynamic=False)
    opt = create_optimizer(base, lr=1.107e-4, weight_decay=5.16e-5, use_muon=False)
    ce = nn.CrossEntropyLoss(ignore_index=255)
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for step in range(STEPS):
        b = next(train_iter)
        x, y = b["image"].to(device).float(), b["label"].to(device).long()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ce(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        if step == 0:
            print(f"  {name}: first step (incl. compile) {time.perf_counter() - t0:.0f}s", flush=True)
        times.append(time.perf_counter() - t0)
        losses.append(float(loss))
        if not torch.isfinite(loss):
            raise SystemExit(f"{name}: non-finite loss at step {step}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    med = statistics.median(times[WARMUP:]) * 1e3
    print(f"  {name}: {STEPS} steps ok  loss {losses[0]:.3f} -> {statistics.mean(losses[-20:]):.3f}  "
          f"median step {med:.1f} ms  peak mem {peak:.1f} GB", flush=True)
    return base, model, med, peak


def check_eval(name, model, val_batches, device):
    model.eval()
    worst, t = 0.0, []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in val_batches:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(b["image"].to(device).float()).float()
            torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
            worst = max(worst, float(out.logsumexp(dim=1).abs().max()))
    print(f"  {name}: eval |logsumexp| max {worst:.2e} (0 = normalized log-probs), "
          f"median {statistics.median(t[1:]) * 1e3:.0f} ms/batch", flush=True)
    model.train()
    return worst


def main():
    device = torch.device("cuda")
    train_iter = iter(loader(augment=True).train)
    val_batches = [b for _, b in zip(range(6), loader(augment=False).val)]
    print(f"{STEPS} bf16 steps per head, batch 4, real cached crops, torch.compile(dynamic=False)\n")

    _, plain, t_plain, m_plain = run("plain 1x1", dict(BASE), train_iter, device)
    check_eval("plain 1x1", plain, val_batches, device)
    del plain
    torch.cuda.empty_cache()

    base, het, t_het, m_het = run("HetSNGP  ", dict(BASE, uncertainty_head=HEAD), train_iter, device)
    lse = check_eval("HetSNGP  ", het, val_batches, device)
    print(f"\n  HetSNGP overhead: step time {100 * (t_het / t_plain - 1):+.0f}%, "
          f"peak memory {m_het - m_plain:+.1f} GB ({m_het:.1f} GB of 40)\n")

    n_fit = 20
    fit_loader = loader(augment=False, batch_size=16)
    stats = fit_laplace_covariance(base, base.decoder.head, fit_loader.train, device=device,
                                   amp_dtype=torch.bfloat16, max_batches=n_fit, log_every=0)
    full = stats["seconds"] / n_fit * (len(fit_loader.train_dataset.records) / 16)
    print(f"  Laplace: {n_fit} batches of 16 in {stats['seconds']:.0f}s -> full train split "
          f"~{full / 60:.0f} min; mean relative posterior variance {stats['mean_relative_posterior_variance']:.3g}")
    lse_fitted = check_eval("HetSNGP fitted", base, val_batches, device)

    ok = lse < 1e-3 and lse_fitted < 1e-3 and m_het < 39.0
    print("\nVERDICT:", "OK -- safe to launch" if ok else "PROBLEM -- see above")


if __name__ == "__main__":
    main()
