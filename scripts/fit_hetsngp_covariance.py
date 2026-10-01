#!/usr/bin/env python3
"""Fit the Laplace posterior covariance of a trained (Het)SNGP checkpoint.

The GP output layer is trained at its posterior mode beta_hat (Algorithm 1 of
Fortuin et al., TMLR 2022); its predictive variance needs the Laplace
covariance Sigma_c = (I + sum_i p_ic (1 - p_ic) Phi_i Phi_i^T)^-1 (Eq. 5). This
script computes it over the training set -- no augmentation, at the
checkpoint's saved (EMA) weights -- and writes it into the checkpoint's
model_state (and ema_state), after which every consumer that loads the
checkpoint (eval_global_miou.py, mars-inference, ...) gets the full HetSNGP
predictive of Algorithm 2.

    python scripts/fit_hetsngp_covariance.py checkpoints/hetsngp_scratch/best_30ep.pt \
        --crop-cache-dir data/.../derived/seg_crop_cache_full
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import torch

from vision_backend.model.hetsngp import HetSNGPHead2d, fit_laplace_covariance
from vision_backend.training.builders import load_segmentation_model_from_checkpoint
from vision_backend.training.utils import load_loader_bundle, select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--loader-config-path", default=None,
                    help="Defaults to the loader config the checkpoint was trained with.")
    ap.add_argument("--crop-cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    ap.add_argument("--max-batches", type=int, default=None, help="Smoke-test cap.")
    ap.add_argument("--output", default=None, help="Defaults to updating the checkpoint in place.")
    args = ap.parse_args()

    device = select_device(torch)
    ckpt_path = Path(args.checkpoint)
    model, model_kind, num_classes, config = load_segmentation_model_from_checkpoint(ckpt_path, device=str(device))
    head = model.decoder.head
    if not (isinstance(head, HetSNGPHead2d) and head.use_gp):
        raise SystemExit(f"{ckpt_path}: decoder.head is not a GP head ({type(head).__name__})")
    print(f"Device: {device}\nHead: {head}")

    loader_cfg_path = args.loader_config_path or config.get("data", {}).get("loader_config_path")
    kwargs = json.loads(Path(loader_cfg_path).read_text())
    kwargs.update(batch_size=args.batch_size, num_workers=args.num_workers, augment=False)
    if args.crop_cache_dir:
        kwargs["cache_dir"] = args.crop_cache_dir
    if model_kind == "context" and config.get("model", {}).get("use_context", True):
        raise SystemExit("context-branch checkpoints are not supported here yet")
    bundle = load_loader_bundle("vision_backend.seg_dataset:create_segmentation_dataloaders", kwargs)
    n_train = len(bundle["train_dataset"].records) if bundle.get("train_dataset") is not None else "?"
    print(f"Laplace data: train split, {n_train} crops, no augmentation, batch {args.batch_size}")

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp_dtype]
    stats = fit_laplace_covariance(model, head, bundle["train"], device=device, amp_dtype=amp,
                                   max_batches=args.max_batches)
    print("Laplace fit:", json.dumps(stats, indent=1))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cov = head.covariance.detach().cpu()
    for key in ("model_state", "ema_state"):
        if key in ckpt:
            ckpt[key]["decoder.head.covariance"] = cov
    ckpt["laplace"] = dict(stats, fitted_at=datetime.datetime.now().isoformat(timespec="seconds"),
                           loader_config_path=str(loader_cfg_path), max_batches=args.max_batches)
    out = Path(args.output) if args.output else ckpt_path
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, out)
    print(f"wrote covariance {tuple(cov.shape)} -> {out}")


if __name__ == "__main__":
    main()
