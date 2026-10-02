#!/usr/bin/env python3
"""Why did the tuning trials run at ~1.2 step/s instead of ~12?

Times, on the GPU: (1) the Lovász-softmax loss alone (fwd+bwd) at the real
batch size -- current batched [N, C] sort along dim 0, a [C, N] row-sort
variant, and the authors' per-class 1-D loop; (2) a full compiled Muon step
with and without drop-path.
"""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_backend.model.optimizers import build_routed_muon_nadam_optimizer
from vision_backend.training.builders import build_simmim_segmentation_model
from vision_backend.training.lovasz import lovasz_softmax

DEV = torch.device("cuda")
B, C, H, W = 4, 14, 512, 512


def timed(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def lovasz_rows(logits, target, ignore_index=255):
    c = logits.shape[1]
    probas = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, c)
    labels = target.reshape(-1)
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    fg = F.one_hot(labels.long(), c).float().t().contiguous()      # [C, N]
    errors = (fg - probas.t()).abs()
    errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
    fg_sorted = fg.gather(1, perm)
    gts = fg_sorted.sum(dim=1, keepdim=True)
    jac = 1.0 - (gts - fg_sorted.cumsum(1)) / (gts + (1.0 - fg_sorted).cumsum(1))
    grad = torch.cat([jac[:, :1], jac[:, 1:] - jac[:, :-1]], dim=1)
    per_class = (errors_sorted * grad).sum(dim=1)
    return per_class[gts.squeeze(1) > 0].mean()


def lovasz_loop(logits, target, ignore_index=255):
    c = logits.shape[1]
    probas = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, c)
    labels = target.reshape(-1)
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    present = torch.bincount(labels, minlength=c).tolist()
    losses = []
    for k in range(c):
        if present[k] == 0:
            continue
        fg = (labels == k).float()
        errors = (fg - probas[:, k]).abs()
        es, perm = torch.sort(errors, descending=True)
        fs = fg[perm]
        g = fs.sum()
        jac = 1.0 - (g - fs.cumsum(0)) / (g + (1.0 - fs).cumsum(0))
        jac = torch.cat([jac[:1], jac[1:] - jac[:-1]])
        losses.append(torch.dot(es, jac))
    return torch.stack(losses).mean()


def bench_lovasz():
    target = torch.randint(0, 13, (B, H, W), device=DEV)
    for name, fn in (("batched [N,C] dim-0 sort (current)", lovasz_softmax),
                     ("row-sort [C,N]", lovasz_rows), ("per-class 1-D loop (reference)", lovasz_loop)):
        logits = torch.randn(B, C, H, W, device=DEV, requires_grad=True)

        def step():
            logits.grad = None
            fn(logits, target).backward()
        print(f"  Lovász {name:36s}: {timed(step):8.1f} ms fwd+bwd", flush=True)
    ref = lovasz_loop(logits.detach(), target)
    print(f"  agreement: current {float(lovasz_softmax(logits.detach(), target)):.6f}  rows "
          f"{float(lovasz_rows(logits.detach(), target)):.6f}  loop {float(ref):.6f}", flush=True)


def bench_step(drop_path):
    torch._dynamo.reset()
    torch.manual_seed(0)
    base = build_simmim_segmentation_model({
        "model_kind": "simmim", "in_channels": 1, "global_base_grid": 32, "window_size": 8,
        "drop_path": drop_path, "decoder_channels": 256, "decoder_dropout": 0.1, "num_classes": 14,
    }).to(DEV).train()
    model = torch.compile(base, dynamic=False)
    opt = build_routed_muon_nadam_optimizer(base, muon_lr=1.1e-4, muon_weight_decay=5e-5, nadam_lr=1.1e-4,
                                            nadam_weight_decay=5e-5, muon_scope="transformer",
                                            muon_lr_mode="match_adam")
    ce = nn.CrossEntropyLoss(ignore_index=255)
    x = torch.randn(B, 1, H, W, device=DEV)
    y = torch.randint(0, 14, (B, H, W), device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ce(model(x), y)
        loss.backward()
        opt.step()
    print(f"  compiled Muon step, drop_path={drop_path}: {timed(step, n=30, warm=5):6.1f} ms", flush=True)


if __name__ == "__main__":
    print(f"batch {B}x{C}x{H}x{W} = {B*H*W:,} pixels\n", flush=True)
    bench_lovasz()
    bench_step(0.0)
    bench_step(0.1)
