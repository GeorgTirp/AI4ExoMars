#!/usr/bin/env python3
"""Fit the campaign-0 Laplace prior: diagonal MC Fisher of the Stage-3
segmenter on NOAH-H training crops (vision_backend/laplace_prior.py).

    python scripts/fit_laplace_prior.py --n-images 2000 \
        --out checkpoints/laplace/noahh_prior_e10.pt \
        --calibrate-with checkpoints/block_finetune/experts_v1_plain/best.pt

--calibrate-with places the tau grid: for a plain (prior-free) block fine-tune
it computes Q = 1/2 sum_j F_j (theta_ft - mu)_j^2, the prior cost per unit tau
of the weight change that run made, and prints the tau for which that cost
equals c x the run's training data loss (N_train x mean CE at its best epoch),
c = 0.1, 1, 10: a weak, balanced and strong prior relative to the expert data.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend.laplace_prior import mc_fisher_diagonal_segmentation, quadratic_form, save_prior  # noqa: E402
from vision_backend.seg_dataset import SegmentationCropDataset, load_seg_records  # noqa: E402
from vision_backend.training.builders import load_segmentation_model_from_checkpoint  # noqa: E402

DER = Path("data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seg-checkpoint", default="checkpoints/arch_ablation_no_lovasz/best_16ep.pt")
    ap.add_argument("--loader-config", default=str(DER / "seg_loader_DC_full.json"))
    ap.add_argument("--crop-cache-dir", default=str(DER / "seg_crop_cache_full"))
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/laplace/noahh_prior_e10.pt")
    ap.add_argument("--calibrate-with", default=None, help="A prior-free block fine-tune checkpoint (best.pt).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    seg, kind, _, _ = load_segmentation_model_from_checkpoint(Path(args.seg_checkpoint), device=str(device))
    cfg = json.loads(Path(args.loader_config).read_text())
    train = [r for r in load_seg_records(cfg["manifest_path"]) if r.split == "train"]
    pick = np.random.default_rng(args.seed).choice(len(train), size=min(args.n_images, len(train)), replace=False)
    ds = SegmentationCropDataset([train[i] for i in sorted(pick)], imagery_path=cfg["imagery_path"],
                                 label_path=cfg["label_path"], augment=False, cache_dir=args.crop_cache_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    t0 = time.time()
    fisher, n = mc_fisher_diagonal_segmentation(seg, loader, device=device)
    mean = {k: p.detach() for k, p in seg.named_parameters()}
    save_prior(args.out, mean, fisher, n_images=n, source_checkpoint=args.seg_checkpoint,
               kind="diag_mc_fisher_per_image", seed=args.seed)
    flat = torch.cat([f.flatten() for f in fisher.values()])
    print(f"[laplace] {n} images in {time.time() - t0:.0f}s -> {args.out}")
    print(f"[laplace] per-image Fisher diagonal: mean {flat.mean():.3e}, median {flat.median():.3e}, "
          f"max {flat.max():.3e}, zero {float((flat == 0).float().mean()):.1%}")
    groups = {}
    for k, f in fisher.items():
        g = ".".join(k.split(".")[:2])
        groups.setdefault(g, []).append(f.flatten())
    for g, fs in sorted(groups.items()):
        v = torch.cat(fs)
        print(f"    {g:<28} {v.numel():>10,} params  mean F {v.mean():.3e}")

    if args.calibrate_with:
        ckpt = torch.load(args.calibrate_with, map_location="cpu", weights_only=False)
        theta = {k[len("seg."):]: v for k, v in ckpt["block_state"].items() if k.startswith("seg.")}
        prior = {"mean": {k: v.cpu() for k, v in mean.items()}, "fisher": {k: v.cpu() for k, v in fisher.items()}}
        with open(Path(args.calibrate_with).with_name("history.csv"), newline="") as fh:
            row = next(r for r in csv.DictReader(fh) if int(r["epoch"]) == int(ckpt["epoch"]))
        manifest = ckpt["args"]["manifest"]
        with open(manifest, newline="") as fh:
            n_train = sum(r["split"] == "train" for r in csv.DictReader(fh))
        data_loss = n_train * float(row["train_loss"])
        calib = {"n_train": n_train, "train_ce_at_best": float(row["train_loss"]), "data_loss": data_loss}
        for scope in ("all", "encoder"):
            q = quadratic_form(theta, prior, scope=scope)
            calib[scope] = {"Q": q, "tau": {str(c): c * data_loss / q for c in (0.1, 1.0, 10.0)}}
            print(f"[laplace] scope {scope}: Q = {q:.4e} for the plain run's weight change; data loss "
                  f"{data_loss:.1f} -> tau for c = 0.1 / 1 / 10: "
                  + " / ".join(f"{t:.3g}" for t in calib[scope]["tau"].values()))
        Path(args.out).with_suffix(".calibration.json").write_text(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
