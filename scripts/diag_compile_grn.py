#!/usr/bin/env python3
"""Can torch.compile be kept for the HybridEncoder, and what does it buy?

diag_nan_hybrid.py showed eager bf16 is clean while compiled bf16 produces
170 non-finite gradients on step 1 and a different forward loss (3.27 vs
3.02) from identical weights and inputs. The suspect is GRN's hand-written
spatial L2 norm, which autocast does not upcast the way it does the standard
normalisation ops. Candidate fixes, each timed, from one seeded initialisation:

  eager            reference numerics and speed
  compile / orig   the failing configuration, for contrast
  compile / fp32   GRN norm computed in fp32 (fixes a bf16-accumulated sum)
  compile / safe   fp32 AND eps inside the sqrt (also removes the 0/0 in the
                   norm's backward at an all-zero channel, which eager PyTorch
                   masks but a compiled decomposition may not)
  compile / nocomp GRN excluded from compilation, rest compiled

A fix is only acceptable if its step-1 loss matches eager's and every gradient
stays finite.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from vision_backend.model import blocks_v2
from vision_backend.model.optimizers import create_optimizer
from scripts.diag_nan_hybrid import build, first_batches, report_grads

ORIG = blocks_v2.GRN.forward


def grn_fp32(self, x):
    gx = torch.norm(x.float(), p=2, dim=(2, 3), keepdim=True)
    nx = (gx / (gx.mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
    return self.gamma * (x * nx) + self.beta + x


def grn_safe(self, x):
    xf = x.float()
    gx = (xf.pow(2).sum(dim=(2, 3), keepdim=True) + self.eps * self.eps).sqrt()
    nx = (gx / (gx.mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
    return self.gamma * (x * nx) + self.beta + x


GRN_IMPLS = {"orig": ORIG, "fp32": grn_fp32, "safe": grn_safe,
             "nocomp": torch.compiler.disable(ORIG)}


def run(label, batches, device, *, compiled, grn, dtype, scaler_on, lr=3e-9):
    blocks_v2.GRN.forward = GRN_IMPLS[grn]
    torch._dynamo.reset()  # never reuse a graph compiled against another GRN
    model = build(False, device)
    if compiled:
        model = torch.compile(model)
    opt = create_optimizer(model, lr=lr, weight_decay=5.16e-5, use_muon=False)
    ce = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.amp.GradScaler("cuda") if scaler_on else None
    losses, bad_steps, times = [], 0, []
    for step, (x, y) in enumerate(batches, 1):
        x, y = x.to(device).float(), y.to(device).long()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = ce(model(x), y)
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(opt)
        _, bad = report_grads(model)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        losses.append(float(loss)); bad_steps += bool(bad)
    steady = times[4:] or times
    finite = all(l == l for l in losses)
    print(f"  {label:26s} step1 loss={losses[0]:.4f}  final loss={losses[-1]:.4f}  "
          f"steps w/ non-finite grads={bad_steps}/{len(batches)}  "
          f"{'OK ' if finite else 'NaN'}  {1000 * sum(steady) / len(steady):7.1f} ms/step", flush=True)


def main():
    device = torch.device("cuda")
    batches = first_batches(12)
    bf16, fp16 = torch.bfloat16, torch.float16
    print("scratch HybridEncoder, 12 real batches, warmup lr; times exclude the first 4 steps\n")
    run("eager bf16", batches, device, compiled=False, grn="orig", dtype=bf16, scaler_on=False)
    run("compile bf16 / GRN orig", batches, device, compiled=True, grn="orig", dtype=bf16, scaler_on=False)
    run("compile bf16 / GRN fp32", batches, device, compiled=True, grn="fp32", dtype=bf16, scaler_on=False)
    run("compile bf16 / GRN safe", batches, device, compiled=True, grn="safe", dtype=bf16, scaler_on=False)
    run("compile bf16 / GRN eager", batches, device, compiled=True, grn="nocomp", dtype=bf16, scaler_on=False)
    run("compile fp16 / GRN orig", batches, device, compiled=True, grn="orig", dtype=fp16, scaler_on=True)


if __name__ == "__main__":
    main()
