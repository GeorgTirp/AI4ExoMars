#!/usr/bin/env python3
"""Fit the uncertainty and explainability files of a Stage-4 block classifier.

Everything that needs a GPU, the expert training blocks or the NOAH-H training
crops is done here, so inference elsewhere (the labelling app) only needs the
checkpoint and these files.

Written next to the checkpoint (<run>/best.pt):
  best.laplace.pt           last-layer Laplace over the block head's `fc`: per-block
                            predictive and epistemic uncertainty (uncertainty/block_laplace.py)
  best.uncertainty.pt       Mahalanobis Gaussians (MahalanobisStats) over the block feature phi
  best.npca.pt              neural-PCA bases per block class over psi_k = fc.weight[k] * phi
                            (schema v2, bases only; the top-activating blocks are ranked
                            on the observation being analysed)
  segmenter.pt              the fine-tuned segmenter in Stage-3 checkpoint format
  segmenter.uncertainty.pt  pixel Mahalanobis Gaussians of its 14 NOAH-H classes
  segmenter.npca.pt         pixel neural-PCA bases of its 14 NOAH-H classes (bases only)
                            -- both fitted on NOAH-H training crops, in the format
                            fit_calibration_artifacts.sh writes for Stage-3 models
  block_features.npz        phi / offset / labels of the train, val and test blocks
                            (--from-features refits the block-level files without a GPU)
  artifacts.json            what each file is, how it was fitted, and checks
and, for the next fine-tuning campaign (cluster use):
  checkpoints/laplace/campaign1_posterior_<run>.pt
                            diagonal posterior tau * F_NOAH-H + N * F_expert at the fine-tuned
                            weights, over all BlockClassifier parameters

    python scripts/fit_block_artifacts.py checkpoints/block_finetune/v2_final_lr3_e12/best.pt
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_block_classifier import brier  # noqa: E402
from vision_backend.block_dataset import DEFAULT_DC_CLASSES_JSON, BlockDataset, load_block_manifest  # noqa: E402
from vision_backend.laplace_prior import mc_fisher_diagonal_blocks, save_prior  # noqa: E402
from vision_backend.model.features import extract_pixel_features, get_classifier_weight_vector  # noqa: E402
from vision_backend.pc_align.neural_pca import compute_block_psi, fit_class_pca, save_gallery  # noqa: E402
from vision_backend.seg_dataset import IGNORE_INDEX, SegmentationCropDataset, load_seg_records  # noqa: E402
from vision_backend.train_stage4_block_finetune import load_block_classifier, make_loader, summarize  # noqa: E402
from vision_backend.training.builders import load_segmentation_model_from_checkpoint  # noqa: E402
from vision_backend.uncertainty.block_laplace import (LastLayerLaplace, fit_temperature,  # noqa: E402
                                                      link_log_probs, save_last_layer_laplace,
                                                      tune_prior_precision)
from vision_backend.uncertainty.malahanobis import (fit_class_gaussians, mahalanobis_distance,  # noqa: E402
                                                   mahalanobis_distance_map, save_stats)

DER = Path("data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/block_finetune/v2_final_lr3_e12/best.pt")
    ap.add_argument("--manifest", default=None, help="default: the manifest the checkpoint was trained on")
    ap.add_argument("--posterior-out", default=None,
                    help="default: checkpoints/laplace/campaign1_posterior_<run>.pt")
    ap.add_argument("--loader-config", default=str(DER / "seg_loader_DC_full.json"))
    ap.add_argument("--crop-cache-dir", default=str(DER / "seg_crop_cache_full"))
    ap.add_argument("--noahh-crops", type=int, default=1500, help="random NOAH-H train crops for the pixel fits")
    ap.add_argument("--noahh-check-crops", type=int, default=150, help="NOAH-H val crops for the site checks")
    ap.add_argument("--pixels-per-crop-per-class", type=int, default=50)
    ap.add_argument("--max-pixels-per-class", type=int, default=5000)
    ap.add_argument("--presence-fraction", type=float, default=0.05)
    ap.add_argument("--max-npca-samples", type=int, default=200)
    ap.add_argument("--min-npca-samples", type=int, default=10)
    ap.add_argument("--n-components", type=int, default=4)
    ap.add_argument("--mc-samples", type=int, default=2000)
    ap.add_argument("--mc-samples-tune", type=int, default=1000, help="MC samples per lambda in the val sweep")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-fisher", action="store_true", help="no campaign posterior (no backward passes)")
    ap.add_argument("--skip-pixel", action="store_true", help="no segmenter export / NOAH-H pixel fits")
    ap.add_argument("--limit-blocks", type=int, default=None,
                    help="smoke test: at most this many blocks per split (half per block size); "
                         "nothing is written to results/")
    ap.add_argument("--from-features", action="store_true",
                    help="refit the block-level files from block_features.npz (no GPU, no imagery)")
    return ap.parse_args()


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    from scipy.stats import rankdata

    positive = positive.astype(bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- block features
@torch.no_grad()
def extract_block_features(model, recs, *, device, batch_size, num_workers, base) -> dict:
    ds = BlockDataset(recs, block_size=base)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, base=base)
    phi, off, lab, idx = [], [], [], []
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p, o = model.block_features(b["image"].to(device), b["valid"].to(device))
        phi.append(p.float().cpu())
        off.append(o.float().cpu())
        lab.append(b["label"])
        idx.append(b["index"])
    order = torch.cat(idx).numpy()
    return {"phi": torch.cat(phi).numpy(), "offset": torch.cat(off).numpy(), "label": torch.cat(lab).numpy(),
            "block_id": np.array([recs[i].block_id for i in order])}


@torch.no_grad()
def block_features_of_images(model, images: list[torch.Tensor], *, device) -> tuple[np.ndarray, np.ndarray]:
    """phi / offset of single [1, 1, H, W] images in [-1, 1] (-1 = nodata)."""
    phi, off = [], []
    for x in images:
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p, o = model.block_features(x, x > -1.0)
        phi.append(p.float().cpu())
        off.append(o.float().cpu())
    return torch.cat(phi).numpy(), torch.cat(off).numpy()


# --------------------------------------------------------------------------- segmenter (pixel level)
def export_segmenter(ckpt: dict, ckpt_path: Path, out: Path, noahh_miou) -> None:
    """The fine-tuned segmenter as a Stage-3 checkpoint (model_state + the source config)."""
    src = torch.load(ckpt["source_checkpoint"], map_location="cpu", weights_only=False)
    config = copy.deepcopy(src.get("config", {}))
    config["model"] = dict(ckpt["seg_model_config"])
    state = {k[len("seg."):]: v for k, v in ckpt["block_state"].items() if k.startswith("seg.")}
    torch.save({"stage": src.get("stage"), "epoch": ckpt.get("epoch"), "model_state": state, "config": config,
                "metrics": {"noahh_val_miou": noahh_miou},
                "provenance": {"kind": "stage4_block_finetune_segmenter", "block_checkpoint": str(ckpt_path),
                               "source_checkpoint": ckpt["source_checkpoint"]}}, str(out))


def fit_pixel_level(seg, run_dir: Path, args, device, rng) -> tuple[tuple[dict, object], list[torch.Tensor]]:
    """Pixel Mahalanobis + pixel neural-PCA bases on random NOAH-H train crops; returns
    ((summary, MahalanobisStats), NOAH-H val crops for the site checks)."""
    cfg = json.loads(Path(args.loader_config).read_text())
    records = load_seg_records(cfg["manifest_path"])
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    pick = sorted(rng.choice(len(train), size=min(args.noahh_crops, len(train)), replace=False))
    ds = SegmentationCropDataset([train[i] for i in pick], imagery_path=cfg["imagery_path"],
                                 label_path=cfg["label_path"], augment=False, cache_dir=args.crop_cache_dir)
    num = int(seg.decoder.head.out_channels)
    w = torch.stack([get_classifier_weight_vector(seg, c) for c in range(num)]).to(device)   # [S, F]
    feats_by_class = {c: [] for c in range(num)}
    counts = {c: 0 for c in range(num)}
    psi_by_class = {c: [] for c in range(num)}
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            feats = extract_pixel_features(seg, sample["image"].unsqueeze(0).to(device))[0]   # [F, H, W]
            pooled = feats.mean(dim=(1, 2))                       # = extract_pooled_features
            flat = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])
            label = sample["label"].reshape(-1).to(device)
            n_valid = int((label != IGNORE_INDEX).sum())
            for c in range(num):
                where = (label == c).nonzero(as_tuple=True)[0]
                if where.numel() == 0:
                    continue
                if counts[c] < args.max_pixels_per_class:
                    take = where[torch.randperm(where.numel(), generator=g)[:args.pixels_per_crop_per_class].to(device)]
                    feats_by_class[c].append(flat[take].cpu())
                    counts[c] += take.numel()
                if n_valid and where.numel() / n_valid >= args.presence_fraction \
                        and len(psi_by_class[c]) < args.max_npca_samples:
                    psi_by_class[c].append((pooled * w[c]).cpu())
            if (i + 1) % 250 == 0:
                print(f"[pixel] {i + 1}/{len(ds)} NOAH-H crops, {time.time() - t0:.0f}s")
    features = torch.cat([torch.cat(v) for v in feats_by_class.values() if v])
    labels = torch.cat([torch.full((sum(t.shape[0] for t in v),), c, dtype=torch.long)
                        for c, v in feats_by_class.items() if v])
    stats = fit_class_gaussians(features, labels, shared_covariance=True)
    save_stats(stats, run_dir / "segmenter.uncertainty.pt")
    bases, npca = {}, {}
    for c, rows in psi_by_class.items():
        if len(rows) < args.min_npca_samples:
            npca[c] = {"n": len(rows), "skipped": True}
            continue
        psi = torch.stack(rows)
        bases[c] = fit_class_pca(psi, class_id=c, n_components=args.n_components)
        npca[c] = {"n": len(rows), "explained_variance_ratio": _explained(psi, bases[c])}
    save_gallery({}, run_dir / "segmenter.npca.pt", bases=bases)
    print(f"[pixel] Mahalanobis: {len(stats.class_stats)} classes, pixels {counts}; "
          f"neural PCA: {len(bases)} classes; {time.time() - t0:.0f}s")

    vpick = sorted(rng.choice(len(val), size=min(args.noahh_check_crops, len(val)), replace=False))
    vds = SegmentationCropDataset([val[i] for i in vpick], imagery_path=cfg["imagery_path"],
                                  label_path=cfg["label_path"], augment=False, cache_dir=args.crop_cache_dir)
    val_images = [vds[i]["image"].unsqueeze(0) for i in range(len(vds))]
    return ({"n_crops": len(ds), "pixels_per_class": counts, "reference_max_distance": stats.reference_max_distance,
             "npca": npca}, stats), val_images


def _explained(psi: torch.Tensor, pca) -> list[float]:
    total = float(psi.float().var(dim=0).sum())
    return [round(float(v) / total, 4) for v in pca.eigvals] if total > 0 else []


@torch.no_grad()
def pixel_distance(seg, stats, images: list[torch.Tensor], device) -> float:
    """Median normalised pixel Mahalanobis distance over the valid pixels of `images`."""
    s = stats.to(device)
    vals = []
    for x in images:
        x = x.to(device)
        d = mahalanobis_distance_map(extract_pixel_features(seg, x), s)[0]
        vals.append((d[x[0, 0] > -1.0] / s.reference_max_distance).cpu())
    return float(torch.cat(vals).median())


def tiles_512(images: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for x in images:
        n = x.shape[-1] // 512
        out += [x[..., r * 512:(r + 1) * 512, c * 512:(c + 1) * 512] for r in range(n) for c in range(n)]
    return out


# --------------------------------------------------------------------------- block level
def fit_block_level(feats: dict, wb: torch.Tensor, classes, run_dir: Path, args, device, extra: dict,
                    link: str = "softmax") -> dict:
    """Last-layer Laplace, block Mahalanobis and block neural-PCA bases, with checks."""
    W, b = wb[:, :-1], wb[:, -1]
    t = {s: {k: torch.from_numpy(v) for k, v in feats[s].items() if k not in ("block_id", "cell")} for s in SPLITS}
    logits = {s: t[s]["offset"] + t[s]["phi"] @ W.T + b for s in SPLITS}
    y = {s: t[s]["label"].long() for s in SPLITS}
    obs = np.array([c.split(":")[0] for c in feats["test"]["cell"]])
    out = {}

    # --- last-layer Laplace
    t0 = time.time()
    la = LastLayerLaplace.fit(t["train"]["phi"].to(device), logits["train"].to(device), wb, link=link,
                              meta={"fitted_on": "train split", "n_blocks": int(len(y["train"]))})
    # lambda is tuned separately for each predictive on its own val NLL; the stored default is the
    # Monte Carlo one (unbiased for any link, and the one that gives mutual information)
    val_args = (logits["val"].to(device), t["val"]["phi"].to(device), y["val"].to(device))
    lam_cf, table_cf = tune_prior_precision(la.to(device), *val_args, method="probit")
    lam, table = tune_prior_precision(la.to(device), *val_args, method="mc", n_samples=args.mc_samples_tune,
                                      seed=args.seed)
    la.prior_precision = lam
    la.meta.update({"prior_precision_tuned_on": "val NLL of the Monte Carlo predictive",
                    "prior_precision_probit": lam_cf})
    save_last_layer_laplace(la, run_dir / "best.laplace.pt")
    edge = {"mc": lam in (table[0][0], table[-1][0]), "probit": lam_cf in (table_cf[0][0], table_cf[-1][0])}
    print(f"[block] last-layer Laplace ({link}): GGN {la.eigvecs.shape[0]} params, lambda* on val: MC {lam:.3g}, "
          f"probit {lam_cf:.3g} (grid edge: {edge}), {time.time() - t0:.0f}s")
    T = fit_temperature(logits["val"], y["val"], link)
    lad = la.to(device)
    g = torch.Generator().manual_seed(args.seed)
    test_phi, test_logits = t["test"]["phi"].to(device), logits["test"].to(device)
    probit = lad.predictive(test_logits, test_phi, method="probit", prior_precision=lam_cf)
    mc = lad.predictive(test_logits, test_phi, method="mc", n_samples=args.mc_samples, generator=g)
    variants = {"map": link_log_probs(logits["test"], link).numpy(),
                "map_temperature": link_log_probs(logits["test"] / T, link).numpy(),
                "laplace_probit": np.log(probit["probs"].cpu().numpy().clip(1e-12)),
                "laplace_mc": np.log(mc["probs"].cpu().numpy().clip(1e-12))}
    yt = y["test"].numpy()
    calib = {}
    for name, z in variants.items():
        m = summarize(z, yt, classes)
        calib[name] = {"accuracy": m["accuracy"], "nll": m["nll"], "ece": m["ece"], "brier": brier(z, yt),
                       "by_obs": {o: {k: summarize(z[obs == o], yt[obs == o], classes)[k] for k in ("nll", "ece")}
                                  for o in sorted(set(obs))}}
    out["last_layer_laplace"] = {"link": link, "prior_precision": lam, "prior_precision_probit": lam_cf,
                                 "lambda_grid_val_nll": {"mc": table, "probit": table_cf},
                                 "lambda_at_grid_edge": edge, "temperature_val": T, "test": calib}

    # --- block Mahalanobis
    stats = fit_class_gaussians(t["train"]["phi"], y["train"], shared_covariance=True)
    save_stats(stats, run_dir / "best.uncertainty.pt")
    dist = {s: mahalanobis_distance(t[s]["phi"], stats).numpy() / stats.reference_max_distance for s in SPLITS}
    wrong = variants["map"].argmax(1) != yt
    pt = np.exp(variants["map_temperature"])
    out["block_mahalanobis"] = {
        "feature": "phi = [LN(z), p_bar]", "feature_dim": int(stats.feature_dim),
        "classes_fitted": sorted(int(c) for c in stats.class_stats),
        "reference_max_distance": stats.reference_max_distance,
        "median_normalised_distance": {"train": float(np.median(dist["train"])), "val": float(np.median(dist["val"])),
                                       **{f"test {o}": float(np.median(dist["test"][obs == o]))
                                          for o in sorted(set(obs))}}}
    if "oxia_phi" in extra:
        d_ox = mahalanobis_distance(torch.from_numpy(extra["oxia_phi"]), stats).numpy() / stats.reference_max_distance
        out["block_mahalanobis"]["median_normalised_distance"]["NOAH-H Oxia val crops (other site)"] = \
            float(np.median(d_ox))
        out["block_mahalanobis"]["auroc_oxia_vs_expert_test"] = auroc(
            np.concatenate([d_ox, dist["test"]]), np.r_[np.ones(len(d_ox)), np.zeros(len(dist["test"]))])
        ox_logits = torch.from_numpy(extra["oxia_offset"] + extra["oxia_phi"] @ W.T.numpy() + b.numpy())
        ox = lad.predictive(ox_logits.to(device), torch.from_numpy(extra["oxia_phi"]).to(device), method="mc",
                            n_samples=args.mc_samples, generator=g)
        out["last_layer_laplace"]["oxia_vs_expert_test"] = {
            "mean_mutual_information": {"NOAH-H Oxia val crops": float(ox["mutual_information"].mean()),
                                        "expert test": float(mc["mutual_information"].mean())},
            "auroc_mutual_information": auroc(np.r_[ox["mutual_information"].cpu().numpy(),
                                                    mc["mutual_information"].cpu().numpy()],
                                              np.r_[np.ones(len(d_ox)), np.zeros(len(yt))])}
    out["misclassification_auroc_test"] = {
        "1 - max p (temperature)": auroc(1 - pt.max(1), wrong),
        "laplace entropy (probit)": auroc(probit["entropy"].cpu().numpy(), wrong),
        "laplace entropy (MC)": auroc(mc["entropy"].cpu().numpy(), wrong),
        "laplace mutual information (MC)": auroc(mc["mutual_information"].cpu().numpy(), wrong),
        "block Mahalanobis distance": auroc(dist["test"], wrong)}

    # --- block neural-PCA bases
    bases, npca = {}, {}
    for k, name in enumerate(classes.names):
        rows = t["train"]["phi"][y["train"] == k]
        if len(rows) < args.min_npca_samples:
            npca[name] = {"class_id": k, "n": int(len(rows)), "skipped": True}
            continue
        psi = compute_block_psi(W, rows, k)
        bases[k] = fit_class_pca(psi, class_id=k, n_components=args.n_components)
        npca[name] = {"class_id": k, "n": int(len(rows)), "explained_variance_ratio": _explained(psi, bases[k])}
    save_gallery({}, run_dir / "best.npca.pt", bases=bases)
    out["block_npca"] = npca
    print(f"[block] Mahalanobis on phi ({stats.feature_dim}-dim, {len(stats.class_stats)} classes); "
          f"neural PCA bases for {len(bases)}/{len(classes.names)} classes")
    return out


# --------------------------------------------------------------------------- campaign posterior
def fit_posterior(model, ckpt: dict, train_recs, args, device, out_path: Path, base: int) -> dict:
    for p in model.parameters():
        p.requires_grad_(True)
    loader = make_loader(BlockDataset(train_recs, block_size=base), batch_size=1, shuffle=False,
                         num_workers=args.num_workers, base=base)
    t0 = time.time()
    fisher, n = mc_fisher_diagonal_blocks(model, loader, device=device)
    cargs = ckpt["args"]
    tau = float(cargs.get("laplace_tau") or 0.0)
    noahh = torch.load(cargs["laplace_prior"], map_location="cpu", weights_only=False) if cargs.get("laplace_prior") \
        else {"fisher": {}}
    mean, precision = {}, {}
    for name, p in model.named_parameters():
        mean[name] = p.detach()
        lam = n * fisher[name].cpu()
        if name.startswith("seg.") and name[4:] in noahh["fisher"]:
            lam = lam + tau * noahh["fisher"][name[4:]]
        precision[name] = lam
    save_prior(out_path, mean, precision, kind="diag_laplace_posterior",
               fisher_expert_mean={k: v.cpu() for k, v in fisher.items()}, n_expert=n, tau_noahh=tau,
               noahh_prior=cargs.get("laplace_prior"), source_checkpoint=args.checkpoint,
               usage="precision is in data units: DiagonalLaplacePrior(path, block_classifier, tau=1.0) over ALL "
                     "BlockClassifier parameters (names 'seg.*' and the head), penalty / N_train as in Stage 4")
    flat = torch.cat([f.flatten() for f in fisher.values()])
    print(f"[posterior] {n} blocks in {time.time() - t0:.0f}s -> {out_path}")
    return {"path": str(out_path), "n_expert_blocks": n, "tau_noahh": tau,
            "expert_fisher_per_block": {"mean": float(flat.mean()), "median": float(flat.median())},
            "seconds": round(time.time() - t0)}


# --------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent
    device = torch.device("cuda" if torch.cuda.is_available() and not args.from_features else "cpu")
    model, ckpt = load_block_classifier(ckpt_path, device=str(device))
    cargs = ckpt["args"]
    manifest = args.manifest or cargs["manifest"]
    records, classes = load_block_manifest(manifest, cargs.get("dc_classes_json") or DEFAULT_DC_CLASSES_JSON,
                                           cargs.get("block_classes"))
    if classes.names != ckpt["classes"]["names"]:
        raise SystemExit("checkpoint class list differs from the manifest's")
    with open(manifest, newline="") as fh:
        cell_of = {r["block_id"]: r.get("cell", r["block_id"]) for r in csv.DictReader(fh)}
    base = int(ckpt["head_config"].get("tile_size", 512))
    splits = {s: [r for r in records if r.split == s] for s in SPLITS}
    if args.limit_blocks:
        with open(manifest, newline="") as fh:
            size_of = {r["block_id"]: r.get("size", "") for r in csv.DictReader(fh)}
        for s in SPLITS:
            by_size = {}
            for r in splits[s]:
                by_size.setdefault(size_of.get(r.block_id, ""), []).append(r)
            per = max(1, args.limit_blocks // max(1, len(by_size)))
            splits[s] = [r for group in by_size.values() for r in group[:per]]
    wb = torch.cat([model.fc.weight.detach(), model.fc.bias.detach()[:, None]], dim=1).float().cpu()
    feat_path = run_dir / "block_features.npz"
    summary = json.loads((run_dir / "artifacts.json").read_text()) if args.from_features and \
        (run_dir / "artifacts.json").is_file() else {}
    extra = {}

    if args.from_features:
        z = np.load(feat_path)
        feats = {s: {k: z[f"{s}_{k}"] for k in ("phi", "offset", "label", "block_id", "cell")} for s in SPLITS}
        if "oxia_phi" in z:
            extra = {"oxia_phi": z["oxia_phi"], "oxia_offset": z["oxia_offset"]}
    else:
        t0 = time.time()
        feats = {}
        for s in SPLITS:
            feats[s] = extract_block_features(model, splits[s], device=device, batch_size=args.batch_size,
                                              num_workers=args.num_workers, base=base)
            feats[s]["cell"] = np.array([cell_of[i] for i in feats[s]["block_id"]])
        print(f"[block] features of {sum(len(v['label']) for v in feats.values())} blocks in {time.time() - t0:.0f}s")

        if not args.skip_pixel:
            seg_path = run_dir / "segmenter.pt"
            ev = run_dir / "eval.json"
            noahh_miou = json.loads(ev.read_text()).get("noahh_val_miou") if ev.is_file() else None
            export_segmenter(ckpt, ckpt_path, seg_path, noahh_miou)
            seg, _, _, _ = load_segmentation_model_from_checkpoint(seg_path, device=str(device))
            with torch.no_grad():
                x = torch.randn(1, 1, 512, 512, device=device)
                diff = float((seg(x) - model.seg(x)).abs().max())
            print(f"[pixel] exported {seg_path}: max |logit difference| to the block model's segmenter {diff:.2e}")
            (pix, pix_stats), noahh_val = fit_pixel_level(seg, run_dir, args, device, rng)
            pix["export_max_logit_difference"] = diff
            # site check: how far from NOAH-H training pixels are other NOAH-H val crops vs the expert sites?
            test_imgs = {}
            for r_i, r in enumerate(splits["test"]):
                o = cell_of[r.block_id].split(":")[0]
                test_imgs.setdefault(o, []).append(r_i)
            tds = BlockDataset(splits["test"], block_size=base)
            pix["median_normalised_pixel_distance"] = {"NOAH-H val crops": pixel_distance(seg, pix_stats, noahh_val, device)}
            for o, idx in sorted(test_imgs.items()):
                pick = sorted(rng.choice(idx, size=min(40, len(idx)), replace=False))
                imgs = tiles_512([tds[i]["image"].unsqueeze(0) for i in pick])
                pix["median_normalised_pixel_distance"][f"expert test {o}"] = pixel_distance(seg, pix_stats, imgs, device)
            summary["segmenter_pixel_level"] = pix
            del seg
            extra["oxia_phi"], extra["oxia_offset"] = block_features_of_images(model, noahh_val, device=device)

        np.savez(feat_path, **{f"{s}_{k}": v for s, d in feats.items() for k, v in d.items()}, fc_weight_bias=wb.numpy(),
                 **extra)

    for s in SPLITS:
        acc = float(((feats[s]["offset"] + feats[s]["phi"] @ wb[:, :-1].T.numpy() + wb[:, -1].numpy()).argmax(1)
                     == feats[s]["label"]).mean())
        summary.setdefault("block_feature_check", {})[f"{s}_accuracy_from_features"] = acc
    print(f"[block] accuracy recomputed from the features: {summary['block_feature_check']}")
    summary["block_level"] = fit_block_level(feats, wb, classes, run_dir, args, device, extra,
                                             link=ckpt["head_config"].get("link", "softmax"))

    if not args.skip_fisher and not args.from_features:
        out = Path(args.posterior_out or f"checkpoints/laplace/campaign1_posterior_{run_dir.name}.pt")
        summary["campaign_posterior"] = fit_posterior(model, ckpt, splits["train"], args, device, out, base)

    summary.update({
        "checkpoint": str(ckpt_path), "manifest": manifest, "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "classes": [{"id": k, "name": n, "noahh_dc_index": s} for k, (n, s) in
                    enumerate(zip(classes.names, classes.seg_index))],
        "preprocessing": {
            "input": "single-band HiRISE RED, raw DN, 0 = nodata, native 0.25 m/px",
            "stretch": "u8 = clip(round(58 + (DN - p1) / (p99 - p1) * (178 - 58)), 1, 255), DN 0 -> 0, with p1/p99 "
                       "the observation's 1st/99th DN percentiles (the labelling export's preprocessing bounds) -- "
                       "scripts/prep_expert_block_manifest.py:to_uint8(stretch='oxia'). Matches the Oxia training "
                       "contrast; the export's own 0-255 stretch has ~2x the contrast and must not be used.",
            "model_input": "x = u8 / 127.5 - 1 as [B, 1, H, W], valid = (u8 != 0); blocks of 512 px, or multiples "
                           "of 512 (tiled through the backbone at 512)"},
        "files": {
            "best.pt": "block classifier: load_block_classifier(path) in vision_backend/train_stage4_block_finetune.py",
            "eval.json": "test metrics; 'temperature' = T* for temperature-scaled probabilities softmax(logits / T)",
            "best.laplace.pt": "last-layer Laplace (uncertainty/block_laplace.py): la = load_last_layer_laplace(path); "
                               "phi, offset = model.block_features(x, valid); logits = offset + model.fc(phi); "
                               "la.predictive(logits, phi, method='probit' | 'mc') -> probs, entropy "
                               "(+ aleatoric, mutual_information with 'mc')",
            "best.uncertainty.pt": "block Mahalanobis (MahalanobisStats over phi): uncertainty/malahanobis.py "
                                   "load_stats + mahalanobis_distance(phi, stats) / stats.reference_max_distance",
            "best.npca.pt": "block neural-PCA bases (schema v2, gallery empty): pc_align/neural_pca.py "
                            "load_gallery_bases(path)[k]; psi = compute_block_psi(model.fc.weight, phi, k); "
                            "rank_indices_by_component(psi, bases[k]) ranks the observation's blocks",
            "segmenter.pt": "the fine-tuned segmenter as a Stage-3 checkpoint (14 NOAH-H pixel classes): "
                            "load_segmentation_model_from_checkpoint(path)",
            "segmenter.uncertainty.pt": "pixel Mahalanobis of segmenter.pt (uncertainty_mapping.epistemic_uncertainty_map)",
            "segmenter.npca.pt": "pixel neural-PCA bases of segmenter.pt (schema v2, gallery empty; "
                                 "psi = compute_psi_batch(seg, k, x))",
            "block_features.npz": "phi / offset / label / block_id / cell per split; refit the block-level files "
                                  "with scripts/fit_block_artifacts.py --from-features"},
    })
    (run_dir / "artifacts.json").write_text(json.dumps(summary, indent=2))
    if not args.limit_blocks:
        res = Path("results/block_artifacts")
        res.mkdir(parents=True, exist_ok=True)
        (res / f"{run_dir.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary["block_level"][k] for k in ("misclassification_auroc_test",)}, indent=2))
    for name, m in summary["block_level"]["last_layer_laplace"]["test"].items():
        print(f"  test {name:<16} acc {m['accuracy']:.4f}  NLL {m['nll']:.3f}  ECE {m['ece']:.3f}  Brier {m['brier']:.3f}")
    print(f"wrote {run_dir / 'artifacts.json'}")


if __name__ == "__main__":
    main()
