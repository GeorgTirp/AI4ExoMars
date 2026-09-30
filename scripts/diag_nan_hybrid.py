#!/usr/bin/env python3
"""Where does the HybridEncoder's first non-finite gradient come from?

Both pretraining-A/B arms produced a NaN loss at batch 3 under bf16 -- with a
finite loss at batches 1-2 and a warmup learning rate of ~1e-9, which is far
too small for any FINITE gradient to wreck the weights. So a gradient must be
non-finite from the first steps. Under fp16 the same runs lasted an epoch,
plausibly because GradScaler.step() silently skips any update whose gradients
contain inf/NaN; bf16 has no scaler, so the first bad gradient lands.

This replays the real first training batches (same loader, seed and crop
cache as stage 3) and, for each precision mode, reports per step: the loss,
whether any gradient is non-finite and in which parameters, and -- in the
anomaly-detection mode -- the forward operation that produced it. It also
reports whether the fp16 GradScaler actually skipped steps, which would
confirm that fp16 was hiding the problem rather than not having it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from vision_backend.seg_dataset import create_segmentation_dataloaders
from vision_backend.training.builders import (
    build_simmim_segmentation_model,
    load_simmim_encoder_checkpoint,
)
from vision_backend.model.optimizers import create_optimizer

DER = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived"
CFG = dict(in_channels=1, num_classes=14, decoder_channels=256, decoder_dropout=0.1,
           global_base_grid=32, window_size=8, drop_path=0.0)


def first_batches(n: int):
    kw = json.loads(Path(f"{DER}/seg_loader_DC_full.json").read_text())
    kw.update(batch_size=4, num_workers=2, seed=42, augment=True,
              cache_dir=f"{DER}/seg_crop_cache_full", persistent_workers=False)
    loader = create_segmentation_dataloaders(**kw).train
    out = []
    for i, b in enumerate(loader):
        out.append((b["image"].clone(), b["label"].clone()))
        if i + 1 == n:
            break
    return out


def build(pretrained: bool, device):
    torch.manual_seed(42)
    model = build_simmim_segmentation_model(CFG)
    if pretrained:
        load_simmim_encoder_checkpoint(torch, model.encoder, "checkpoints/stage1_simmim/last.pt",
                                       prefer_ema=True, strict=True)
    return model.to(device).train()


def report_grads(model) -> tuple[float, list[str]]:
    bad, sq = [], 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
        else:
            sq += float(p.grad.float().pow(2).sum())
    return math.sqrt(sq), bad


def run(mode: str, batches, device, pretrained: bool, lr: float):
    model = build(pretrained, device)
    if mode.startswith("compile"):
        model = torch.compile(model)
    opt = create_optimizer(model, lr=lr, weight_decay=5.16e-5, use_muon=False)
    ce = nn.CrossEntropyLoss(ignore_index=255)
    dtype = torch.bfloat16 if "bf16" in mode else torch.float16
    scaler = torch.amp.GradScaler("cuda") if "fp16" in mode else None
    print(f"\n--- {mode}  ({'pretrained' if pretrained else 'scratch'}, lr={lr:g}) ---", flush=True)
    for step, (x, y) in enumerate(batches, 1):
        x, y = x.to(device).float(), y.to(device).long()
        opt.zero_grad(set_to_none=True)
        try:
            with torch.autograd.set_detect_anomaly("anomaly" in mode):
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model(x)
                    loss = ce(logits, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
        except RuntimeError as exc:
            msg = str(exc).splitlines()[0]
            print(f"  step {step}: ANOMALY -> {msg[:300]}", flush=True)
            return
        if scaler is not None:
            scaler.unscale_(opt)
        gnorm, bad = report_grads(model)
        act_max = float(logits.detach().float().abs().max())
        line = (f"  step {step}: loss={float(loss):.4f}  grad_norm={gnorm:.3e}  "
                f"logits|max|={act_max:.3e}  nonfinite_grads={len(bad)}")
        if bad:
            line += f"  e.g. {bad[:3]}"
        if scaler is not None:
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            line += f"  scaler {before:.0f}->{scaler.get_scale():.0f}" + \
                    ("  [STEP SKIPPED]" if scaler.get_scale() < before else "")
        else:
            opt.step()
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    # Warmup LR at step 1 of the real runs: 1.107e-4 / 40,359 warmup steps.
    ap.add_argument("--lr", type=float, default=3e-9)
    args = ap.parse_args()
    device = torch.device("cuda")
    batches = first_batches(args.steps)
    print(f"replaying {len(batches)} real training batches of shape {tuple(batches[0][0].shape)}")
    for pretrained in (False, True):
        for mode in ("eager-fp16", "eager-bf16", "eager-bf16-anomaly", "compile-bf16"):
            run(mode, batches, device, pretrained, args.lr)


if __name__ == "__main__":
    main()
