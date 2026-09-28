#!/usr/bin/env python3
"""Can the Stage-3 segmentation model memorise ONE batch?

This is the sharpest available split between "the model/loss is broken" and
"the training configuration is wrong". A network with enough capacity must be
able to drive cross-entropy on a single fixed batch to near zero by brute
memorisation -- no generalisation required. 28.8M parameters against 8 crops is
overwhelming capacity.

Background: the v0 sweep run spent 36 h and ~647,000 steps moving train_loss
2.359 -> 2.301, and a 1%-subset overfit run plateaued at 2.33 with val_miou
frozen from epoch 4. 2.30 is ln(10): the model is emitting the class prior and
ignoring its input entirely.

Three arms, each from an identical initialisation and the same fixed batch, so
any difference is attributable to the arm and nothing else:

  plain    AdamW(lr, wd=0), unweighted CE, no IG head, no dropout
           -- the control. If THIS cannot fit one batch, the defect is in the
           model or the loss, and no optimizer change will save it.
  project  the repo's own create_optimizer + class weights + IG aux loss,
           exactly what run_variant_sweep.sh trains with.
  nodecay  project, but weight_decay=0 -- isolates the two optimizer defects
           found by inspection (no decay/no-decay split, and NAdam's
           decoupled_weight_decay defaulting to False, making it coupled L2).

Reported per arm: loss, the gradient norm reaching the encoder stem (is signal
flowing back at all?), and how many distinct classes the argmax emits (a
collapsed model predicts exactly one).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vision_backend.seg_dataset import (
    SegmentationCropDataset,
    load_seg_records,
    partition_records,
)
from vision_backend.training.builders import build_context_segmentation_model
from vision_backend.model.optimizers import create_optimizer
from vision_backend.training.utils import compute_class_weights

V0 = dict(
    in_channels=1, local_base_channels=52, context_base_channels=26,
    context_dim=256, use_stage32=True, swin_depths=(2, 2, 2),
    swin_num_heads=(4, 8, 16), window_size=8, drop_path=0.0,
    num_classes=14, decoder_channels=256, use_context=False,
)


def get_batch(args, device):
    der = Path(args.der)
    records = load_seg_records(der / "seg_crops_DC_full.csv")
    train, _ = partition_records(records)
    ds = SegmentationCropDataset(
        train[: args.batch], imagery_path=der / "drg_on_label_grid.tif",
        label_path=der / "labels_DC_classid.tif",
        cache_dir=der / "seg_crop_cache_full",
        augment=False, spatial_jitter_px=0,
    )
    xs = torch.stack([ds[i]["image"] for i in range(len(ds))]).to(device)
    ys = torch.stack([ds[i]["label"] for i in range(len(ds))]).to(device)
    return xs, ys


def run_arm(name, xs, ys, device, *, steps, lr, seed, weighted, use_ig, wd, project_opt):
    torch.manual_seed(seed)
    cfg = dict(V0)
    if use_ig:
        cfg["num_classes_ig"] = 5
    model = build_context_segmentation_model(cfg).to(device)
    model.train()

    weight = None
    if weighted:
        counts = torch.bincount(
            ys[ys != 255].reshape(-1), minlength=V0["num_classes"]
        ).tolist()
        weight = compute_class_weights(
            torch, counts, V0["num_classes"], scheme="inverse_sqrt", clip_max=10.0
        ).to(device)
    ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)

    if project_opt:
        opt = create_optimizer(model, lr=lr, weight_decay=wd, use_muon=False, llrd=1.0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    stem = next(p for n, p in model.named_parameters() if p.ndim == 4)
    trace = []
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(xs)
        logits = out[0] if isinstance(out, tuple) else out
        loss = ce(logits, ys)
        if step % max(steps // 10, 1) == 0:
            with torch.no_grad():
                n_classes = int(logits.argmax(1).unique().numel())
            g = stem.grad.norm().item() if stem.grad is not None else float("nan")
            trace.append((step, round(loss.item(), 4), round(g, 6), n_classes))
            print(f"  [{name}] step {step:4d}  loss={loss.item():.4f} "
                  f"stem_grad={g:.3e}  distinct_pred_classes={n_classes}", flush=True)
        loss.backward()
        opt.step()
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--der", default="data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/diag/overfit_one_batch.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = get_batch(args, device)
    valid = (ys != 255)
    print(f"device={device}  batch={tuple(xs.shape)}  valid_px={valid.float().mean():.3f}")
    print(f"label classes present: {sorted(ys[valid].unique().tolist())}")
    print(f"image range: [{xs.min():.3f}, {xs.max():.3f}]  std={xs.std():.4f}\n")

    arms = {
        "plain":   dict(weighted=False, use_ig=False, wd=0.0,  project_opt=False),
        "project": dict(weighted=True,  use_ig=True,  wd=1e-2, project_opt=True),
        "nodecay": dict(weighted=True,  use_ig=True,  wd=0.0,  project_opt=True),
    }
    results = {}
    for name, kw in arms.items():
        print(f"--- arm: {name} ---")
        results[name] = run_arm(
            name, xs, ys, device, steps=args.steps, lr=args.lr, seed=args.seed, **kw
        )
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")

    print("\n=== VERDICT ===")
    for name, tr in results.items():
        first, last = tr[0][1], tr[-1][1]
        print(f"{name:8s} loss {first:.3f} -> {last:.3f}  "
              f"(final distinct predicted classes: {tr[-1][3]})")
    print("\nAn arm that cannot drive a single batch's loss toward 0 with 28.8M "
          "params has a model/loss defect, not a tuning problem.")


if __name__ == "__main__":
    main()
