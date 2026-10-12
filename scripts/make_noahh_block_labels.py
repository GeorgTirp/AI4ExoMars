#!/usr/bin/env python3
"""Block labels from the NOAH-H rasters, in the Stage-4 manifest format.

A stand-in for the planetary scientists' block labels, to dry-run
train_stage4_block_finetune.py end to end before they arrive: each 512 block
of the segmentation manifest gets its dominant NOAH-H DC class (blocks whose
dominant class covers < --min-frac of the labelled pixels are dropped, as an
expert would not give them a single class). The segmentation train/val split
is kept (val is the spatially separate east strip).

--as-new relabels the given DC ids as "NEW: <name>", so the run treats them as
classes without a segmenter prior -- exercising the new-class path the real
labels will need.

    python scripts/make_noahh_block_labels.py --out job_outputs/block_dryrun/blocks.csv \
        --per-split 4000 1000 --as-new 3 7 10 11
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DER = Path("data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=str(DER / "seg_crops_DC_full.csv"))
    ap.add_argument("--crop-cache-dir", default=str(DER / "seg_crop_cache_full"))
    ap.add_argument("--classes-json", default=str(DER / "classes_DC.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-split", type=int, nargs=2, default=(4000, 1000), metavar=("TRAIN", "VAL"),
                    help="Blocks sampled per split before the --min-frac filter.")
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--as-new", type=int, nargs="*", default=[], help="DC ids to relabel as new classes.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = {int(c["id"]): c["name"] for c in json.loads(Path(args.classes_json).read_text())["classes"]}
    with open(args.manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads((Path(args.crop_cache_dir) / "meta.json").read_text())
    if int(meta["count"]) != len(rows):
        raise SystemExit(f"crop cache has {meta['count']} crops, manifest {len(rows)} rows")
    labels = np.load(Path(args.crop_cache_dir) / "labels.npy", mmap_mode="r")
    m, S = int(meta["jitter_margin"]), int(meta["crop_size"])

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for split, n in zip(("train", "val"), args.per_split):
        idx = np.array([i for i, r in enumerate(rows) if r["split"] == split])
        idx = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
        kept = 0
        for i in idx:
            lab = np.asarray(labels[i, m:m + S, m:m + S])
            counts = np.bincount(lab[lab > 0].ravel(), minlength=max(names) + 1)
            if counts.sum() < 0.9 * S * S:
                continue
            dc = int(counts.argmax())
            frac = counts[dc] / counts.sum()
            if frac < args.min_frac:
                continue
            label = f"NEW: {names[dc]}" if dc in args.as_new else names[dc]
            out_rows.append({"col": rows[i]["col"], "row": rows[i]["row"], "label": label,
                             "split": split, "dominant_frac": f"{frac:.3f}"})
            kept += 1
        print(f"{split}: {kept}/{len(idx)} blocks with a dominant class >= {args.min_frac}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["col", "row", "label", "split", "dominant_frac"])
        writer.writeheader()
        writer.writerows(out_rows)
    hist = {}
    for r in out_rows:
        hist[(r["split"], r["label"])] = hist.get((r["split"], r["label"]), 0) + 1
    for (split, label), c in sorted(hist.items()):
        print(f"  {split:<5} {label:<60} {c}")
    print(f"wrote {len(out_rows)} blocks to {out}")


if __name__ == "__main__":
    main()
