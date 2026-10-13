#!/usr/bin/env python3
"""Evaluate a Stage-4 block classifier: calibration, uncertainty of the
estimate, overfitting checks, and forgetting of the NOAH-H segmentation.

1. Temperature scaling (Guo et al., "On Calibration of Modern Neural
   Networks", ICML 2017): one scalar T fitted on the val logits by NLL,
   applied to test. Accuracy / F1 are unchanged by T; NLL, Brier and ECE are
   reported raw and calibrated.
2. 95 % confidence intervals for the test metrics from a spatial bootstrap:
   whole cells (the 1 km split cells) are resampled, not single blocks,
   because neighbouring blocks are correlated.
3. Overfitting of the model selection: val vs test on the SAME class set (the
   classes with >= --min-support blocks in both splits).
4. Forgetting (--noahh-eval): the fine-tuned segmenter, exported as a Stage-3
   checkpoint, scored on the NOAH-H val crops with scripts/eval_global_miou.py
   -- the same evaluation as the Stage-3 runs (E10 reference: 0.2072).

    python scripts/eval_block_classifier.py checkpoints/block_finetune/<run>/best.pt \
        --manifest data/BlockLabels_prepared/manifest.csv --noahh-eval
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend.block_dataset import BlockDataset, load_block_manifest  # noqa: E402
from vision_backend.train_stage4_block_finetune import load_block_classifier, make_loader, predict, summarize  # noqa: E402

DER = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived"


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """argmin_T NLL(logits / T) on a log-spaced grid, refined once."""
    def nll(t):
        return -log_softmax(logits / t)[np.arange(len(labels)), labels].mean()
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    t0 = grid[np.argmin([nll(t) for t in grid])]
    fine = np.exp(np.linspace(np.log(t0 / 1.05), np.log(t0 * 1.05), 200))
    return float(fine[np.argmin([nll(t) for t in fine])])


def brier(logits: np.ndarray, labels: np.ndarray) -> float:
    p = np.exp(log_softmax(logits))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(labels)), labels] = 1
    return float(((p - onehot) ** 2).sum(1).mean())


def bootstrap_ci(logits, labels, groups, classes, *, n_boot: int, seed: int, min_support: int) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "macro_f1_sel", "nll", "ece")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        m = summarize(logits[idx], labels[idx], classes, min_support=min_support)
        for k in keys:
            draws[k].append(m[k])
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
            for k, v in draws.items()}


def same_class_f1(metrics: dict, keep: set[int]) -> float:
    vals = [c["f1"] for k, c in enumerate(metrics["per_class"]) if k in keep and c["f1"] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--block-classes", default=None, help="default: block_classes.json next to the manifest")
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--noahh-eval", action="store_true", help="Also score the segmenter on the NOAH-H val crops.")
    ap.add_argument("--out", default=None, help="JSON output (default: eval.json next to the checkpoint)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out) if args.out else ckpt_path.with_name("eval.json")
    block_classes = args.block_classes or (Path(args.manifest).parent / "block_classes.json")
    block_classes = block_classes if Path(block_classes).is_file() else None
    records, classes = load_block_manifest(args.manifest, block_classes=block_classes)
    with open(args.manifest, newline="") as fh:
        cell_of = {r["block_id"]: r.get("cell", r["block_id"]) for r in csv.DictReader(fh)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_block_classifier(ckpt_path, device=str(device))
    if ckpt["classes"]["names"] != classes.names:
        raise SystemExit("checkpoint class list differs from the manifest's")
    base = int(ckpt["head_config"].get("tile_size", 512))

    out = {"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"), "manifest": args.manifest}
    preds = {}
    for split in ("val", "test"):
        recs = [r for r in records if r.split == split]
        ds = BlockDataset(recs, block_size=base)
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, base=base)
        # predict() follows the loader's batch order; recover each block's record via "index"
        logits, labels, order = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    z = model(batch["image"].to(device), batch["valid"].to(device))
                logits.append(z.float().cpu().numpy())
                labels.append(batch["label"].numpy())
                order.append(batch["index"].numpy())
        idx = np.concatenate(order)
        preds[split] = (np.concatenate(logits), np.concatenate(labels),
                        np.array([cell_of[recs[i].block_id] for i in idx]))

    zv, yv, _ = preds["val"]
    zt, yt, gt = preds["test"]
    T = fit_temperature(zv, yv)
    # per-block predictions, for paired comparisons between runs (scripts/compare_block_runs.py)
    np.savez(out_path.with_name("eval_preds.npz"), temperature=T,
             **{f"{s}_{k}": v for s, (z, y, g) in preds.items() for k, v in (("logits", z), ("labels", y), ("cells", g))})
    raw = summarize(zt, yt, classes, min_support=args.min_support)
    cal = summarize(zt / T, yt, classes, min_support=args.min_support)
    val = summarize(zv, yv, classes, min_support=args.min_support)
    out["temperature"] = T
    out["test_raw"] = raw
    out["test_calibrated"] = {k: cal[k] for k in ("nll", "ece")} | {"brier": brier(zt / T, yt)}
    out["test_raw"]["brier"] = brier(zt, yt)
    out["val_calibrated_nll"] = summarize(zv / T, yv, classes)["nll"]
    out["test_ci95"] = bootstrap_ci(zt / T, yt, gt, classes, n_boot=args.n_boot, seed=0, min_support=args.min_support)
    sv = np.bincount(yv, minlength=len(classes.names))
    st = np.bincount(yt, minlength=len(classes.names))
    keep = {k for k in range(len(classes.names)) if sv[k] >= args.min_support and st[k] >= args.min_support}
    out["same_class_set"] = {"classes": sorted(classes.names[k] for k in keep),
                             "val_macro_f1": same_class_f1(val, keep), "test_macro_f1": same_class_f1(raw, keep)}

    print(f"checkpoint {ckpt_path} (epoch {ckpt.get('epoch')}), temperature T* = {T:.3f} (fitted on val)")
    print(f"test: acc {raw['accuracy']:.4f} {out['test_ci95']['accuracy']}  macro-F1 {raw['macro_f1']:.4f} "
          f"{out['test_ci95']['macro_f1']}  macro-F1(>= {args.min_support}) {raw['macro_f1_sel']:.4f} "
          f"{out['test_ci95']['macro_f1_sel']}")
    print(f"test NLL {raw['nll']:.3f} -> {cal['nll']:.3f}   ECE {raw['ece']:.3f} -> {cal['ece']:.3f}   "
          f"Brier {out['test_raw']['brier']:.3f} -> {out['test_calibrated']['brier']:.3f}   (raw -> calibrated)")
    s = out["same_class_set"]
    print(f"val vs test macro-F1 on the same {len(keep)} classes: {s['val_macro_f1']:.4f} vs {s['test_macro_f1']:.4f}")

    if args.noahh_eval:
        seg_state = {k[len("seg."):]: v for k, v in ckpt["block_state"].items() if k.startswith("seg.")}
        seg_path = ckpt_path.with_name("segmenter_stage3_format.pt")
        torch.save({"model_state": seg_state, "config": {"model": ckpt["seg_model_config"]}}, seg_path)
        json_out = ckpt_path.with_name("noahh_val_miou.json")
        subprocess.run([sys.executable, "scripts/eval_global_miou.py", str(seg_path), "--batch-size", "8",
                        "--loader-config-path", f"{DER}/seg_loader_DC_full.json",
                        "--crop-cache-dir", f"{DER}/seg_crop_cache_full", "--json-out", str(json_out)], check=True)
        out["noahh_val_miou"] = json.loads(json_out.read_text())[0]["global_miou"]
        print(f"NOAH-H val mIoU of the fine-tuned segmenter: {out['noahh_val_miou']}  (before fine-tuning: 0.2072)")
        seg_path.unlink()

    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
