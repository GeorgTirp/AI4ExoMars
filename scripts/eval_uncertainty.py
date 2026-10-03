#!/usr/bin/env python3
"""Calibration and uncertainty quality of segmentation checkpoints.

Per checkpoint, on the 512 val set split into two interleaved halves (even /
odd batches -- val is unshuffled, so both halves cover the same areas):

  half A  fits a softmax temperature T* by grid search on pixel NLL
          (temperature scaling, Guo et al. 2017 -- the standard cheap
          calibration baseline any principled method has to beat);
  half B  reports, raw (T = 1) and temperature-scaled (T*):
            mIoU, pixel accuracy,
            NLL      mean -log p(y) over labelled pixels,
            Brier    mean ||p - onehot(y)||^2,
            ECE      15-bin top-label expected calibration error,
            AUROC    misclassification detection from predictive entropy
                     (and from max-probability), i.e. does high uncertainty
                     flag the wrong pixels,
          plus, for HetSNGP heads, the misclassification AUROC of the two
          uncertainty components on their own: GP variance (model /
          distance) and heteroscedastic variance (data).

All statistics are accumulated over every labelled pixel (histograms for
ECE/AUROC), never on a subsample. Writes --json-out.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from vision_backend.model.features import hook_pre_classifier_features
from vision_backend.model.hetsngp import HetSNGPHead2d
from vision_backend.training.builders import load_segmentation_model_from_checkpoint
from vision_backend.training.utils import load_loader_bundle, parse_segmentation_batch, select_device

TEMPS = torch.exp(torch.linspace(math.log(0.4), math.log(6.0), 49))
N_ECE, N_HIST = 15, 4000


class Acc:
    """Pixel statistics for one (model, temperature) on half B."""

    def __init__(self, c, device):
        self.c = c
        self.conf = torch.zeros(c, c, dtype=torch.float64, device=device)
        self.nll = torch.zeros((), dtype=torch.float64, device=device)
        self.brier = torch.zeros((), dtype=torch.float64, device=device)
        self.n = 0
        self.ece = torch.zeros(3, N_ECE, dtype=torch.float64, device=device)  # count, sum conf, sum correct
        self.hist = {}  # score name -> [2, N_HIST] (correct, wrong)
        self.ranges = {}

    def add(self, logp, y):  # logp [N, C] normalized log-probs, y [N]
        p = logp.exp()
        pred = p.argmax(1)
        correct = pred == y
        self.conf += torch.bincount(y * self.c + pred, minlength=self.c * self.c).view(self.c, self.c).double()
        self.nll += -logp.gather(1, y[:, None]).sum().double()
        self.brier += ((p - F.one_hot(y, self.c).float()) ** 2).sum().double()
        self.n += y.numel()
        top = p.max(1).values
        b = (top * N_ECE).long().clamp(max=N_ECE - 1)
        self.ece[0] += torch.bincount(b, minlength=N_ECE).double()
        self.ece[1] += torch.bincount(b, weights=top.double(), minlength=N_ECE)
        self.ece[2] += torch.bincount(b, weights=correct.double(), minlength=N_ECE)
        self.add_score("entropy", -(p * logp).sum(1), correct, (0.0, math.log(self.c)))
        self.add_score("1-maxprob", 1.0 - top, correct, (0.0, 1.0))

    def add_score(self, name, score, correct, rng):
        lo, hi = rng
        b = ((score - lo) / (hi - lo) * N_HIST).long().clamp(0, N_HIST - 1)
        h = self.hist.setdefault(name, torch.zeros(2, N_HIST, dtype=torch.float64, device=score.device))
        h[0] += torch.bincount(b[correct], minlength=N_HIST).double()
        h[1] += torch.bincount(b[~correct], minlength=N_HIST).double()

    def summary(self):
        inter = self.conf.diagonal()
        union = self.conf.sum(0) + self.conf.sum(1) - inter
        present = union > 0
        cnt, sconf, scorr = self.ece
        ece = float((cnt / cnt.sum() * ((sconf - scorr).abs() / cnt.clamp(min=1))).sum())
        out = {"miou": float((inter[present] / union[present]).mean()),
               "pixel_acc": float(inter.sum() / self.conf.sum()),
               "nll": float(self.nll / self.n), "brier": float(self.brier / self.n), "ece": ece}
        for name, h in self.hist.items():
            out[f"auroc_{name}"] = auroc(h)
        return out


def auroc(h):
    """P(score_wrong > score_correct) from two histograms over the same bins (ties = 1/2)."""
    corr, wrong = h[0], h[1]
    below_corr = torch.cumsum(corr, 0) - corr  # correct pixels in lower bins
    num = (wrong * (below_corr + 0.5 * corr)).sum()
    return float(num / (corr.sum() * wrong.sum()))


def flat(logp, y, ignore):
    logp = logp.permute(0, 2, 3, 1).reshape(-1, logp.shape[1])
    y = y.reshape(-1)
    keep = y != ignore
    return logp[keep], y[keep], keep


def evaluate(path, loader, device, ignore, max_batches=None):
    model, kind, c, cfg = load_segmentation_model_from_checkpoint(path, device=str(device))
    head = model.decoder.head
    het = isinstance(head, HetSNGPHead2d)
    nll_grid = torch.zeros(len(TEMPS), dtype=torch.float64, device=device)
    raw = Acc(c, device)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y, ctx = parse_segmentation_batch(batch)
            x, y = x.to(device).float(), y.to(device).long()
            # fp32, as the training loop's validation pass scores these models
            with hook_pre_classifier_features(model) as cap:
                out = model(x) if ctx is None else model(x, ctx.to(device).float())
            logp = F.log_softmax(out.float(), 1)
            lp, yy, keep = flat(logp, y, ignore)
            if i % 2 == 0:  # half A: temperature grid
                for t, temp in enumerate(TEMPS.tolist()):
                    nll_grid[t] += -F.log_softmax(lp / temp, 1).gather(1, yy[:, None]).sum().double()
                continue
            raw.add(lp, yy)
            if het:  # uncertainty components, upsampled to the label grid
                maps = head.uncertainty_maps(cap["features"])
                correct = lp.argmax(1) == yy
                for name, m in maps.items():
                    up = F.interpolate(m[:, None], size=y.shape[-2:], mode="bilinear", align_corners=False)
                    s = torch.log10(up.reshape(-1)[keep].clamp_min(1e-12))
                    raw.add_score(f"{name}(log10)", s, correct, (-12.0, 2.0))
    t_star = float(TEMPS[int(nll_grid.argmin())])
    scaled = Acc(c, device)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            if i % 2 == 0:
                continue
            x, y, ctx = parse_segmentation_batch(batch)
            x, y = x.to(device).float(), y.to(device).long()
            out = model(x) if ctx is None else model(x, ctx.to(device).float())
            lp, yy, _ = flat(F.log_softmax(out.float(), 1), y, ignore)
            scaled.add(F.log_softmax(lp / t_star, 1), yy)
    return {"checkpoint": str(path), "head": type(head).__name__, "T_star": t_star,
            "raw": raw.summary(), "temp_scaled": scaled.summary()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--loader-config-path", required=True)
    ap.add_argument("--crop-cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--max-batches", type=int, default=None, help="Smoke-test cap.")
    args = ap.parse_args()

    device = select_device(torch)
    kw = json.loads(Path(args.loader_config_path).read_text())
    kw.update(batch_size=args.batch_size, num_workers=args.num_workers, augment=False)
    if args.crop_cache_dir:
        kw["cache_dir"] = args.crop_cache_dir
    loader = load_loader_bundle("vision_backend.seg_dataset:create_segmentation_dataloaders", kw)["val"]
    rows = []
    for path in args.checkpoints:
        r = evaluate(Path(path), loader, device, args.ignore_index, args.max_batches)
        rows.append(r)
        print(f"\n=== {Path(path).parent.name}/{Path(path).name}  ({r['head']}, T* = {r['T_star']:.2f})", flush=True)
        for tag in ("raw", "temp_scaled"):
            s = r[tag]
            extra = "  ".join(f"{k[6:]} {v:.3f}" for k, v in s.items() if k.startswith("auroc_"))
            print(f"  {tag:11s} mIoU {s['miou']:.4f}  acc {s['pixel_acc']:.4f}  NLL {s['nll']:.4f}  "
                  f"Brier {s['brier']:.4f}  ECE {s['ece']:.4f}  AUROC: {extra}", flush=True)
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
