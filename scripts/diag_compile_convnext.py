#!/usr/bin/env python3
"""Did torch.compile silently skip updates in the v0-v3 comparison?

The HybridEncoder turned out to produce non-finite gradients when compiled,
and under fp16 the GradScaler hid it by skipping every affected update while
the loss stayed finite. The v0-v3 ConvNeXt-Swin variants contain no GRN, but
they were also trained compiled + fp16 + GradScaler, so the same *mechanism*
has to be ruled out directly rather than by inference.

For each variant this replays the same real training batches (with context
crops for v1/v3) from one seeded initialisation, eager and compiled, both fp16
with a GradScaler exactly as trained, and counts steps whose gradients are
non-finite -- i.e. updates the scaler would skip.

Healthy looks like eager: a few skipped steps at the start while the scaler
calibrates its scale down from 65,536, then none. A compiled run that keeps
skipping after calibration would mean the comparison trained on an unknown
fraction of its steps.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from scripts.diag_nan_hybrid import report_grads
from vision_backend.model.optimizers import create_optimizer
from vision_backend.seg_dataset import create_segmentation_dataloaders
from vision_backend.training.builders import build_context_segmentation_model

DER = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived"
BASE = dict(in_channels=1, use_stage32=True, swin_depths=(2, 2, 2), swin_num_heads=(4, 8, 16),
            window_size=8, drop_path=0.0, num_classes=14, decoder_dropout=0.1)
BIG = dict(local_base_channels=52, context_base_channels=26, context_dim=256, decoder_channels=256)
SMALL = dict(local_base_channels=44, context_base_channels=22, context_dim=217, decoder_channels=176)
VARIANTS = {"v0": {**BIG, "use_context": False}, "v1": {**BIG, "use_context": True},
            "v2": {**SMALL, "use_context": False}, "v3": {**SMALL, "use_context": True}}
STEPS, CALIBRATION = 30, 10


def real_batches(n):
    kw = json.loads(Path(f"{DER}/seg_loader_DC_full.json").read_text())
    kw.update(batch_size=4, num_workers=2, seed=42, augment=True, persistent_workers=False,
              cache_dir=f"{DER}/seg_crop_cache_full", use_context=True,
              context_cache_dir=f"{DER}/seg_context_cache_full")
    out = []
    for i, b in enumerate(create_segmentation_dataloaders(**kw).train):
        out.append((b["image"].clone(), b["label"].clone(), b["context"].clone()))
        if i + 1 == n:
            break
    return out


def run(name, cfg, batches, device, compiled):
    torch._dynamo.reset()
    torch.manual_seed(42)
    model = build_context_segmentation_model({**BASE, **cfg}).to(device).train()
    if compiled:
        model = torch.compile(model)
    opt = create_optimizer(model, lr=3e-9, weight_decay=5.16e-5, use_muon=False)
    ce = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.amp.GradScaler("cuda")
    skipped = []
    for x, y, c in batches:
        x, y, c = x.to(device).float(), y.to(device).long(), c.to(device).float()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, c) if cfg["use_context"] else model(x)
            loss = ce(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        _, bad = report_grads(model)
        skipped.append(bool(bad))
        scaler.step(opt)
        scaler.update()
    early, late = sum(skipped[:CALIBRATION]), sum(skipped[CALIBRATION:])
    print(f"  {name:3s} {'compiled' if compiled else 'eager   '}  skipped: first {CALIBRATION} "
          f"steps {early:2d}   steps {CALIBRATION + 1}-{STEPS} {late:2d}/{STEPS - CALIBRATION}   "
          f"final scale {scaler.get_scale():>7.0f}  {'<-- PERSISTENT' if late else ''}", flush=True)
    return late


def main():
    device = torch.device("cuda")
    batches = real_batches(STEPS)
    print(f"{STEPS} real training batches, fp16 + GradScaler as in the comparison runs\n")
    persistent = 0
    for name, cfg in VARIANTS.items():
        for compiled in (False, True):
            persistent += run(name, cfg, batches, device, compiled)
    print("\nVERDICT:", "compiled runs keep skipping updates after calibration -- comparison "
          "suspect" if persistent else "no skipped updates after scaler calibration in any "
          "variant, eager or compiled -- the comparison trained on every step")


if __name__ == "__main__":
    main()
