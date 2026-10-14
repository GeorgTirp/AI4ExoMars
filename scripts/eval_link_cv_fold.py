#!/usr/bin/env python3
"""Uncertainty evaluation of one spatial-CV fold model (E18: softmax vs normcdf head).

For a block classifier trained with --cv-fold f:
- block features of its training blocks and of the held-out fold;
- a last-layer Laplace (uncertainty/block_laplace.py) and block Mahalanobis
  Gaussians fitted on the training blocks;
- the held-out fold scored by cross-fitting: its 1 km cells are split at random
  into two halves; temperature and the Laplace prior precision (separately for
  the closed-form and the Monte Carlo predictive) are tuned on one half and
  applied to the other, then the halves swap -- no label is used both to tune
  and to score;
- 150 NOAH-H Oxia val crops, run as blocks, as a different-site set.

Writes <out>.npz (per held-out block: id, label, cell, the class probabilities
of every predictive and the uncertainty scores; the Oxia scores per half) and
<out>.json (tuned values). scripts/summarize_link_cv.py pools the folds.

    python scripts/eval_link_cv_fold.py checkpoints/block_link_cv/normcdf/fold0/best.pt --fold 0 \
        --out results/block_link_cv/normcdf.fold0
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_block_artifacts import block_features_of_images, extract_block_features  # noqa: E402
from vision_backend.block_dataset import DEFAULT_DC_CLASSES_JSON, load_block_manifest  # noqa: E402
from vision_backend.seg_dataset import SegmentationCropDataset, load_seg_records  # noqa: E402
from vision_backend.train_stage4_block_finetune import cv_splits, load_block_classifier  # noqa: E402
from vision_backend.uncertainty.block_laplace import (LastLayerLaplace, fit_temperature,  # noqa: E402
                                                      link_log_probs, tune_prior_precision)
from vision_backend.uncertainty.malahanobis import fit_class_gaussians, mahalanobis_distance  # noqa: E402

DER = Path("data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived")


def oxia_images(n: int, loader_config: str, cache_dir: str, seed: int) -> list[torch.Tensor]:
    if n <= 0:
        return []
    cfg = json.loads(Path(loader_config).read_text())
    val = [r for r in load_seg_records(cfg["manifest_path"]) if r.split == "val"]
    pick = sorted(np.random.default_rng(seed).choice(len(val), size=min(n, len(val)), replace=False))
    ds = SegmentationCropDataset([val[i] for i in pick], imagery_path=cfg["imagery_path"],
                                 label_path=cfg["label_path"], augment=False, cache_dir=cache_dir)
    return [ds[i]["image"].unsqueeze(0) for i in range(len(ds))]


def scores(la: LastLayerLaplace, logits, phi, *, T: float, lam_cf: float, lam_mc: float, n_mc: int, seed: int,
           stats) -> dict:
    """Class probabilities of each predictive and per-block uncertainty scores."""
    link = la.link
    jv = la.jacobian_basis(phi)
    out = {"p_map": link_log_probs(logits, link).exp(), "p_map_temperature": link_log_probs(logits / T, link).exp()}
    cf = la.predictive(logits, jv=jv, method="probit", prior_precision=lam_cf)
    mc = la.predictive(logits, jv=jv, method="mc", prior_precision=lam_mc, n_samples=n_mc,
                       generator=torch.Generator().manual_seed(seed))
    out |= {"p_laplace_closed": cf["probs"], "p_laplace_mc": mc["probs"],
            "s_1_minus_max_p_temperature": 1 - out["p_map_temperature"].max(1).values,
            "s_entropy_temperature": -(out["p_map_temperature"] * out["p_map_temperature"].clamp_min(1e-12).log()).sum(1),
            "s_entropy_closed": cf["entropy"], "s_entropy_mc": mc["entropy"],
            "s_aleatoric_mc": mc["aleatoric"], "s_mutual_information_mc": mc["mutual_information"],
            "s_mahalanobis": mahalanobis_distance(phi.cpu(), stats) / stats.reference_max_distance}
    if link == "normcdf":
        d = la.predictive(logits, jv=jv, method="dirichlet", prior_precision=lam_cf)
        out |= {"s_entropy_dirichlet": d["entropy"], "s_aleatoric_dirichlet": d["aleatoric"],
                "s_mutual_information_dirichlet": d["mutual_information"]}
    return {k: v.detach().double().cpu().numpy() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", required=True, help="output stem (writes <out>.npz and <out>.json)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--oxia-crops", type=int, default=150)
    ap.add_argument("--loader-config", default=str(DER / "seg_loader_DC_full.json"))
    ap.add_argument("--crop-cache-dir", default=str(DER / "seg_crop_cache_full"))
    ap.add_argument("--mc-samples", type=int, default=2000)
    ap.add_argument("--mc-samples-tune", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_block_classifier(args.checkpoint, device=str(device))
    link = ckpt["head_config"].get("link", "softmax")
    cargs = ckpt["args"]
    manifest = args.manifest or cargs["manifest"]
    records, classes = load_block_manifest(manifest, cargs.get("dc_classes_json") or DEFAULT_DC_CLASSES_JSON,
                                           cargs.get("block_classes"))
    with open(manifest, newline="") as fh:
        cell_of = {r["block_id"]: r.get("cell", r["block_id"]) for r in csv.DictReader(fh)}
    sp = cv_splits(records, args.fold)
    base = int(ckpt["head_config"].get("tile_size", 512))
    feats = {s: extract_block_features(model, sp[s], device=device, batch_size=args.batch_size,
                                       num_workers=args.num_workers, base=base) for s in ("train", "val")}
    wb = torch.cat([model.fc.weight.detach(), model.fc.bias.detach()[:, None]], dim=1).float().cpu()
    W, b = wb[:, :-1], wb[:, -1]

    def tensors(f):
        phi = torch.from_numpy(f["phi"])
        return phi, torch.from_numpy(f["offset"]) + phi @ W.T + b

    phi_tr, logit_tr = tensors(feats["train"])
    phi_va, logit_va = tensors(feats["val"])
    y_tr, y_va = torch.from_numpy(feats["train"]["label"]).long(), torch.from_numpy(feats["val"]["label"]).long()
    ox = oxia_images(args.oxia_crops, args.loader_config, args.crop_cache_dir, args.seed)
    if ox:
        ox_phi, ox_off = (torch.from_numpy(a) for a in block_features_of_images(model, ox, device=device))
        ox_logit = ox_off + ox_phi @ W.T + b

    la = LastLayerLaplace.fit(phi_tr.to(device), logit_tr.to(device), wb, link=link)
    stats = fit_class_gaussians(phi_tr, y_tr, shared_covariance=True)

    cells = np.array([cell_of[i] for i in feats["val"]["block_id"]])
    uniq = np.unique(cells)
    first = set(np.random.default_rng(args.seed).permutation(uniq)[: len(uniq) // 2])
    in_a = np.array([c in first for c in cells])
    n = len(cells)
    res: dict[str, np.ndarray] = {}
    tuned = []
    for half, (cal, ev) in enumerate(((in_a, ~in_a), (~in_a, in_a))):
        cal_t, ev_t = torch.from_numpy(cal), torch.from_numpy(ev)
        T = fit_temperature(logit_va[cal_t], y_va[cal_t], link)
        cal_args = (logit_va[cal_t].to(device), phi_va[cal_t].to(device), y_va[cal_t].to(device))
        lam_cf, tab_cf = tune_prior_precision(la, *cal_args, method="probit")
        lam_mc, tab_mc = tune_prior_precision(la, *cal_args, method="mc", n_samples=args.mc_samples_tune,
                                              seed=args.seed)
        kw = dict(T=T, lam_cf=lam_cf, lam_mc=lam_mc, n_mc=args.mc_samples, seed=args.seed, stats=stats)
        s = scores(la, logit_va[ev_t].to(device), phi_va[ev_t].to(device), **kw)
        for k, v in s.items():
            res.setdefault(k, np.zeros((n,) + v.shape[1:]))[ev] = v
        if ox:
            for k, v in scores(la, ox_logit.to(device), ox_phi.to(device), **kw).items():
                if k.startswith("s_"):
                    res[f"oxia_{k}_half{half}"] = v
        res[f"eval_half{half}"] = ev
        grid_edge = {"probit": lam_cf in (tab_cf[0][0], tab_cf[-1][0]), "mc": lam_mc in (tab_mc[0][0], tab_mc[-1][0])}
        tuned.append({"temperature": T, "lambda_closed": lam_cf, "lambda_mc": lam_mc, "grid_edge": grid_edge,
                      "n_calibration_blocks": int(cal.sum()), "n_eval_blocks": int(ev.sum())})
        print(f"[fold {args.fold} half {half}] T {T:.3f}, lambda closed {lam_cf:.3g}, MC {lam_mc:.3g} "
              f"(edge {grid_edge}); {int(cal.sum())} calibration / {int(ev.sum())} scored blocks")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{args.out}.npz", block_id=feats["val"]["block_id"], label=feats["val"]["label"], cell=cells,
             logits=logit_va.numpy(), **res)
    acc_map = float((logit_va.argmax(1) == y_va).float().mean())
    Path(f"{args.out}.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "fold": args.fold, "link": link, "n_train": len(y_tr), "n_heldout": n,
        "n_oxia": len(ox), "heldout_accuracy_map": acc_map, "halves": tuned, "classes": classes.names,
        "seconds": round(time.time() - t0)}, indent=2))
    print(f"[fold {args.fold}] {link}: held-out accuracy {acc_map:.4f} on {n} blocks; wrote {args.out}.npz "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
