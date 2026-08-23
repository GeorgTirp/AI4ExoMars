#!/usr/bin/env python3
"""Reassign the train/val split of a seg-crop manifest, without rebuilding crops.

The original manifest (``prep_seg_crops.py``) holds out a contiguous right-hand
column band. That is the leak-free choice, but on this mosaic it makes val a
geologically different strip: "Textured non-bedrock" is 9.7% of train pixels and
38.2% of val pixels, so val mIoU measures generalisation to unseen terrain
rather than in-distribution accuracy, and reads far below a comparable
published number.

This rewrites *only* the ``split`` column, preserving manifest row order, so the
padded crop cache built by ``prep_seg_crop_cache.py`` (indexed by that order)
stays valid -- no 37 GB rebuild.

Modes
-----
random
    Per-crop Bernoulli draw. Train and val distributions match closely, but
    adjacent tiles land on both sides, so terrain autocorrelation leaks and val
    scores come out optimistic. Use to answer "is the model learning at all",
    not to report a final number.
blocks (default)
    Assigns whole square super-blocks of ``--block-tiles``^2 tiles to val.
    Keeps distributions close while forcing val terrain to be spatially
    separated from train -- the compromise between the two extremes.
column
    Reproduces the original contiguous right-hand band.

Per-crop class histograms are computed once from the crop cache and memoised to
``<manifest>.crop_hists.npy``, so any later split can be scored instantly (and
so class weights can be summed per-split without another pass over the cache).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

NUM_CLASSES_DEFAULT = 14
IGNORE = 255


def load_rows(manifest: Path) -> tuple[list[dict], list[str]]:
    with manifest.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def crop_histograms(rows, cache_dir: Path, manifest: Path, num_classes: int) -> np.ndarray:
    """(N, num_classes) int64 per-crop pixel counts, memoised next to the manifest."""
    hist_path = manifest.with_suffix(".crop_hists.npy")
    meta_path = manifest.with_suffix(".crop_hists.json")
    key = {"n": len(rows), "num_classes": num_classes,
           "manifest_mtime": manifest.stat().st_mtime}
    if hist_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == key:
            print(f"[resplit] loaded cached per-crop histograms {hist_path.name}")
            return np.load(hist_path)

    meta = json.loads((cache_dir / "meta.json").read_text())
    J, S = int(meta["jitter_margin"]), int(meta["crop_size"])
    imgs = np.load(cache_dir / "images.npy", mmap_mode="r")
    labs = np.load(cache_dir / "labels.npy", mmap_mode="r")
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    out = np.zeros((len(rows), num_classes), dtype=np.int64)
    it = enumerate(rows)
    if tqdm:
        it = tqdm(it, total=len(rows), desc="per-crop histograms", unit="crop")
    for i, _ in it:
        a = imgs[i, J:J + S, J:J + S]
        l = labs[i, J:J + S, J:J + S]
        t = l.astype(np.int64) - 1
        t[l == 0] = IGNORE
        t[a == 0] = IGNORE
        v = (t >= 0) & (t < num_classes)
        if v.any():
            out[i] = np.bincount(t[v], minlength=num_classes)[:num_classes]
    np.save(hist_path, out)
    meta_path.write_text(json.dumps(key))
    print(f"[resplit] wrote per-crop histograms -> {hist_path.name}")
    return out


def assign(rows, mode: str, val_fraction: float, seed: int, block_tiles: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(rows)
    col = np.array([int(r["col"]) for r in rows])
    row = np.array([int(r["row"]) for r in rows])

    if mode == "random":
        return rng.random(n) < val_fraction

    if mode == "column":
        return col + np.array([int(r["size"]) for r in rows]) // 2 >= (
            col.min() + (col.max() - col.min()) * (1.0 - val_fraction)
        )

    # blocks: group tiles into square super-blocks, assign whole blocks to val
    size = int(rows[0]["size"])
    span = size * block_tiles
    bx = (col - col.min()) // span
    by = (row - row.min()) // span
    bid = by * (bx.max() + 1) + bx
    uniq = np.unique(bid)
    rng.shuffle(uniq)
    # take blocks until the crop-count target is met (blocks vary in occupancy)
    target = val_fraction * n
    counts = {b: int((bid == b).sum()) for b in uniq}
    chosen, acc = set(), 0
    for b in uniq:
        if acc >= target:
            break
        chosen.add(b)
        acc += counts[b]
    return np.isin(bid, list(chosen))


def report(hists: np.ndarray, is_val: np.ndarray, num_classes: int) -> None:
    tr = hists[~is_val].sum(0).astype(float)
    va = hists[is_val].sum(0).astype(float)
    tp, vp = tr.sum(), va.sum()
    print(f"\n  crops : train {(~is_val).sum():>7,}   val {is_val.sum():>7,} "
          f"({100*is_val.mean():.1f}%)")
    print(f"  pixels: train {int(tp):>15,}   val {int(vp):>15,}")
    print(f"\n  {'cls':>4} {'train %':>9} {'val %':>9} {'ratio':>8}")
    worst = 0.0
    for c in range(num_classes):
        a = 100 * tr[c] / tp if tp else 0.0
        b = 100 * va[c] / vp if vp else 0.0
        if a == 0 and b == 0:
            continue
        ratio = (b / a) if a > 0 else float("inf")
        worst = max(worst, abs(np.log2(ratio)) if ratio not in (0, float("inf")) else 8.0)
        print(f"  {c:>4} {a:>9.3f} {b:>9.3f} {ratio:>8.2f}")
    print(f"\n  worst class imbalance: {2**worst:.2f}x  (1.00x = identical distributions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--mode", choices=("blocks", "random", "column"), default="blocks")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--block-tiles", type=int, default=8,
                    help="blocks mode: super-block edge in tiles (8 = 8x8 crops)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES_DEFAULT)
    ap.add_argument("--out", default=None, help="default: rewrite manifest in place (.bak kept)")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows, fields = load_rows(manifest)
    hists = crop_histograms(rows, Path(args.cache_dir), manifest, args.num_classes)

    print(f"\n=== CURRENT split ({manifest.name}) ===")
    report(hists, np.array([r.get("split") == "val" for r in rows]), args.num_classes)

    is_val = assign(rows, args.mode, args.val_fraction, args.seed, args.block_tiles)
    print(f"\n=== PROPOSED split (mode={args.mode}, seed={args.seed}"
          + (f", block_tiles={args.block_tiles}" if args.mode == "blocks" else "") + ") ===")
    report(hists, is_val, args.num_classes)

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    out = Path(args.out) if args.out else manifest
    if out == manifest:
        shutil.copy2(manifest, manifest.with_suffix(manifest.suffix + ".bak"))
        print(f"\n[resplit] backed up -> {manifest.name}.bak")
    for r, v in zip(rows, is_val):
        r["split"] = "val" if v else "train"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[resplit] wrote {out}  (row order preserved -> crop cache still valid)")


if __name__ == "__main__":
    main()
