#!/usr/bin/env python3
"""Expert block labels (mars-labeler "mars-block-crops" exports) -> Stage-4 manifest.

    python scripts/prep_expert_block_manifest.py            # all 512-block exports under data/BlockLabels

For every labelled block of every export folder:
- verifies the crop against integrity.json (SHA-256);
- converts it to a uint8 PNG quantised like the Oxia DRG the segmenter was
  trained on: the observation-wide 1st/99th percentiles that the export records
  (dataset.json preprocessing.bounds) are mapped linearly onto the Oxia
  training imagery's 1st/99th percentiles (OXIA_P1_P99), raw 0 -> 0 = nodata.
  The labeler's own 1-99 % -> 0-255 stretch (--stretch labeler) gives about
  twice the Oxia contrast (std ~48 vs ~25);
- skips exports whose block_size is not --block-size (the 1024 ESP_027201
  session of 2026-10 was not meant for this campaign);
- maps the class through the export's `model_index` (the configured channel of
  the existing 14-class segmenter): model_index < 14 -> NOAH-H DC id, written
  as the label "1".."14"; anything else is a new class, labelled by its name.
  block_classes.json fixes the class list -- the 14 NOAH-H classes, then every
  new class of classes.json, present in this campaign or not -- so the block
  head keeps the same outputs across labelling campaigns.

Splits are spatial (README of the export: split by observation or separated
regions): blocks are grouped into --cell-px cells per observation, cells go
to train / val / test at --fractions per observation, and of --trials random
assignments the one covering the most classes is kept (each class scores 2 for
train, 1 each for val and test, minus 10 x the fraction deviation). With
--buffer, train blocks touching a val/test block become "buffer" (kept in the
manifest, never trained or evaluated on). The split is computed from labels.csv
for every labelled block -- crop present or not -- so it does not change when
missing crops arrive later.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Valid-pixel 1st / 99th percentiles of the Oxia DRG training crops (60 crops
# spread over seg_crop_cache_full, measured 2026-10-12; mean 119.3, std 24.9).
OXIA_P1_P99 = (58.0, 178.0)
DC_CLASSES_JSON = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/classes_DC.json"
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blur_valid(u8: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass of the valid pixels only (normalised convolution, so
    nodata edges do not bleed in); nodata stays 0."""
    from scipy.ndimage import gaussian_filter

    valid = u8 > 0
    num = gaussian_filter(np.where(valid, u8, 0).astype(np.float64), sigma)
    den = gaussian_filter(valid.astype(np.float64), sigma)
    out = np.clip(np.rint(num / np.maximum(den, 1e-6)), 1, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def to_uint8(raw: np.ndarray, preprocessing: dict, stretch: str, ref=OXIA_P1_P99) -> np.ndarray:
    method = preprocessing.get("method")
    if method != "global_percentile_uint8":
        if raw.dtype == np.uint8:
            return raw.copy()  # already 8-bit (0 = nodata)
        raise ValueError(f"unsupported preprocessing method {method!r} for {raw.dtype} crops")
    lo, hi = (float(v) for v in preprocessing["bounds"])
    unit = (raw.astype(np.float64) - lo) / (hi - lo)
    v = ref[0] + unit * (ref[1] - ref[0]) if stretch == "oxia" else unit * 255.0
    out = np.clip(np.rint(v), 1, 255).astype(np.uint8)  # 1..255: 0 stays reserved for nodata
    out[raw == 0] = 0
    return out


def load_export(folder: Path) -> dict:
    meta = json.loads((folder / "dataset.json").read_text())
    if meta.get("format") != "mars-block-crops":
        raise SystemExit(f"{folder}: not a mars-block-crops export ({meta.get('format')!r})")
    if meta.get("target_kind") != "whole_block_majority_class":
        raise SystemExit(f"{folder}: target_kind {meta.get('target_kind')!r}, expected whole_block_majority_class")
    with (folder / "labels.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {"folder": folder, "meta": meta, "rows": rows,
            "classes": json.loads((folder / "classes.json").read_text())["classes"],
            "integrity": json.loads((folder / "integrity.json").read_text())["files"]}


def assign_splits(blocks: list[dict], *, fractions, cell_px: int, trials: int, seed: int, buffer: bool):
    n = len(blocks)
    frac = np.asarray(fractions, dtype=float)
    labels = [b["label"] for b in blocks]
    cells = [(b["obs"], (b["x"] + b["size"] // 2) // cell_px, (b["y"] + b["size"] // 2) // cell_px) for b in blocks]
    cell_keys = sorted(set(cells))
    cell_index = {c: i for i, c in enumerate(cell_keys)}
    cell_of = np.array([cell_index[c] for c in cells])
    members = [np.flatnonzero(cell_of == i) for i in range(len(cell_keys))]
    obs_names = sorted({b["obs"] for b in blocks})
    obs_cells = {o: [i for i, c in enumerate(cell_keys) if c[0] == o] for o in obs_names}
    pos = {(b["obs"], b["x"], b["y"]): i for i, b in enumerate(blocks)}
    neighbours = [[pos[k] for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy)
                   and (k := (b["obs"], b["x"] + dx * b["size"], b["y"] + dy * b["size"])) in pos]
                  for b in blocks]
    classes = sorted(set(labels))
    cls = np.array([classes.index(lab) for lab in labels])

    rng = np.random.default_rng(seed)
    best, best_score = None, -np.inf
    for _ in range(trials):
        split = np.zeros(n, dtype=int)
        deviation = 0.0
        for o in obs_names:
            total = sum(len(members[c]) for c in obs_cells[o])
            counts = np.zeros(3)
            for c in rng.permutation(obs_cells[o]):
                s = int(np.argmax(frac * total - counts))  # split furthest below its target
                split[members[c]] = s
                counts[s] += len(members[c])
            deviation += np.abs(counts / total - frac).sum()
        final = split.copy()
        if buffer:
            for i in np.flatnonzero(split == 0):
                if any(split[j] != 0 for j in neighbours[i]):
                    final[i] = 3
        present = np.zeros((len(classes), 4), dtype=bool)
        present[cls, final] = True
        score = (2 * present[:, 0] + present[:, 1] + present[:, 2]).sum() - 10 * deviation
        if score > best_score:
            best, best_score = final, score
    names = (*SPLITS, "buffer")
    return [names[s] for s in best], [f"{c[0]}:{c[1]}:{c[2]}" for c in cells]


def assign_cv_folds(blocks: list[dict], cells: list[str], *, k: int, trials: int, seed: int):
    """Spatial k-fold assignment of whole cells (per observation, balanced in
    size, the assignment covering the most classes in every fold), and for
    each block the folds it borders: when fold f is held out, training blocks
    touching a fold-f block are dropped (same buffer rule as the split)."""
    n = len(blocks)
    cell_keys = sorted(set(cells))
    members = {c: np.array([i for i in range(n) if cells[i] == c]) for c in cell_keys}
    obs_cells = {}
    for c in cell_keys:
        obs_cells.setdefault(c.split(":")[0], []).append(c)
    classes = sorted({b["label"] for b in blocks})
    cls = np.array([classes.index(b["label"]) for b in blocks])
    frac = np.full(k, 1.0 / k)
    rng = np.random.default_rng(seed)
    best, best_score = None, -np.inf
    for _ in range(trials):
        fold = np.zeros(n, dtype=int)
        deviation = 0.0
        for o, oc in sorted(obs_cells.items()):
            total = sum(len(members[c]) for c in oc)
            counts = np.zeros(k)
            for j in rng.permutation(len(oc)):
                f = int(np.argmax(frac * total - counts))
                fold[members[oc[j]]] = f
                counts[f] += len(members[oc[j]])
            deviation += np.abs(counts / total - frac).sum()
        present = np.zeros((len(classes), k), dtype=bool)
        present[cls, fold] = True
        score = present.sum() - 10 * deviation
        if score > best_score:
            best, best_score = fold, score
    pos = {(b["obs"], b["x"], b["y"]): i for i, b in enumerate(blocks)}
    borders = []
    for i, b in enumerate(blocks):
        near = {int(best[pos[key]]) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy)
                and (key := (b["obs"], b["x"] + dx * b["size"], b["y"] + dy * b["size"])) in pos}
        borders.append(sorted(near - {int(best[i])}))
    return [int(f) for f in best], borders


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exports", nargs="*", default=None,
                    help="Export folders (default: every dataset-* folder under data/BlockLabels).")
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[512],
                    help="Use only exports with these block sizes (blocks above 512 are classified whole: "
                         "the backbone sees 512 tiles, the head pools over the whole block).")
    ap.add_argument("--keep-splits-from", default=None,
                    help="Earlier manifest: its blocks keep their split and cell (so test sets stay comparable "
                         "across manifest versions); only the other blocks are assigned.")
    ap.add_argument("--quarter-train", action="store_true",
                    help="Ablation: TRAIN blocks above 512 are written as four 512 quarters with the parent's "
                         "label (val/test stay whole). Default: whole blocks.")
    ap.add_argument("--reuse-from", default=None,
                    help="Earlier output dir with the same --stretch/--blur-sigma: whole-block crops already "
                         "converted there are copied instead of re-converted (checksums still verified).")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="Also assign the train+val blocks to k spatial cross-validation folds (columns fold, "
                         "cv_buffer) for hyperparameter tuning without touching test.")
    ap.add_argument("--out-dir", default="data/BlockLabels_prepared")
    ap.add_argument("--dc-classes-json", default=DC_CLASSES_JSON)
    ap.add_argument("--stretch", choices=("oxia", "labeler"), default="oxia")
    ap.add_argument("--blur-sigma", type=float, default=0.0,
                    help="Gaussian low-pass (px) after the stretch, to match the smoother Oxia training "
                         "imagery (scripts/diag_site_shift.py: sigma 0.7 matches its finest-scale power "
                         "and noise, 1.0 its mid-band power and gradient energy). 0 = off.")
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--cell-px", type=int, default=4096, help="Spatial split cell side (px; 4096 = 1 km).")
    ap.add_argument("--no-buffer", dest="buffer", action="store_false")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true",
                    help="Write the manifest from the crops that are present (default: fail if any is missing).")
    args = ap.parse_args()

    folders = ([Path(f) for f in args.exports] if args.exports else
               sorted(p.parent for p in Path("data/BlockLabels").rglob("dataset-*/dataset.json")))
    exports = []
    for f in folders:
        e = load_export(f)
        if int(e["meta"]["block_size"]) not in args.block_sizes:
            print(f"skipping {f}: block_size {e['meta']['block_size']} not in {args.block_sizes}")
            continue
        exports.append(e)
    if not exports:
        raise SystemExit(f"no export folders with block_size in {args.block_sizes} found")
    for e in exports:
        if e["meta"].get("label_origin") != "labeling_session":
            raise SystemExit(f"{e['folder']}: label_origin {e['meta'].get('label_origin')!r} -- may contain "
                             "unreviewed model predictions; not training on it.")

    # --- class mapping via model_index -----------------------------------------
    key = [(c["id"], c["name"], c["model_index"]) for c in exports[0]["classes"]]
    for e in exports[1:]:
        if [(c["id"], c["name"], c["model_index"]) for c in e["classes"]] != key:
            raise SystemExit(f"{e['folder']}: classes.json differs from {exports[0]['folder']}")
    dc = sorted(json.loads(Path(args.dc_classes_json).read_text())["classes"], key=lambda c: int(c["id"]))
    num_seg = len(dc)
    by_id = {int(c["id"]): c for c in exports[0]["classes"]}
    print(f"Class mapping (export model_index -> segmenter channel; {num_seg} NOAH-H classes):")
    for c in exports[0]["classes"]:
        mi = int(c["model_index"])
        target = f"DC{mi + 1:>2} {dc[mi]['name']}" if mi < num_seg else "NEW"
        print(f"  {c['name']:<45} -> {target}")

    # --- blocks + spatial split (all labelled blocks) ---------------------------
    blocks = []
    for e in exports:
        size = int(e["meta"]["block_size"])
        for r in e["rows"]:
            c = by_id[int(r["class_id"])]
            mi = int(c["model_index"])
            blocks.append({"obs": r["obs_id"], "x": int(r["x_px"]), "y": int(r["y_px"]), "size": size,
                           "label": str(mi + 1) if mi < num_seg else c["name"], "expert_class": c["name"],
                           "block_id": r["block_id"], "edit_count": r["edit_count"],
                           "crop": e["folder"] / "crops" / r["crop_filename"],
                           "sha256": e["integrity"].get(f"crops/{r['crop_filename']}"),
                           "w": int(r["w_px"]), "h": int(r["h_px"]), "export": e})
    kept = {}
    if args.keep_splits_from:
        with open(args.keep_splits_from, newline="") as fh:
            kept = {r["block_id"]: (r["split"], r["cell"]) for r in csv.DictReader(fh)}
    new = [i for i, b in enumerate(blocks) if b["block_id"] not in kept]
    new_splits, new_cells = (assign_splits([blocks[i] for i in new], fractions=args.fractions,
                                           cell_px=args.cell_px, trials=args.trials, seed=args.seed,
                                           buffer=args.buffer) if new else ([], []))
    kept_cells = {c for _, c in kept.values()}
    if kept_cells & set(new_cells):
        raise SystemExit("new blocks fall into cells of the kept split -- extending an already split "
                         "observation needs cell-level split inheritance, not implemented")
    assigned = dict(zip(new, zip(new_splits, new_cells)))
    splits = [kept[b["block_id"]][0] if b["block_id"] in kept else assigned[i][0] for i, b in enumerate(blocks)]
    cells = [kept[b["block_id"]][1] if b["block_id"] in kept else assigned[i][1] for i, b in enumerate(blocks)]
    if kept:
        print(f"kept the split of {len(blocks) - len(new)} blocks from {args.keep_splits_from}; "
              f"assigned {len(new)} new blocks")
    folds, borders = [None] * len(blocks), [[] for _ in blocks]
    if args.cv_folds:
        cv = [i for i, s in enumerate(splits) if s in ("train", "val")]
        f_cv, b_cv = assign_cv_folds([blocks[i] for i in cv], [cells[i] for i in cv], k=args.cv_folds,
                                     trials=args.trials, seed=args.seed)
        for i, f, bd in zip(cv, f_cv, b_cv):
            folds[i], borders[i] = f, bd

    missing = [b for b in blocks if not b["crop"].is_file()]
    if missing:
        per_obs = Counter(b["obs"] for b in missing)
        msg = ", ".join(f"{o}: {n} of {sum(b['obs'] == o for b in blocks)}" for o, n in sorted(per_obs.items()))
        if not args.allow_missing:
            raise SystemExit(f"crops missing ({msg}) -- complete the upload, or pass --allow-missing")
        print(f"WARNING: crops missing ({msg}); writing the manifest from the present ones")

    # --- convert crops -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    rows, stats = [], {}
    reuse = Path(args.reuse_from) if args.reuse_from else None
    if reuse is not None:
        prev = json.loads((reuse / "summary.json").read_text())
        if (prev.get("stretch"), float(prev.get("blur_sigma", 0.0))) != (args.stretch, args.blur_sigma):
            raise SystemExit(f"--reuse-from {reuse}: converted with different --stretch/--blur-sigma")
    for b, split, cell, fold, border in zip(blocks, splits, cells, folds, borders):
        if not b["crop"].is_file():
            continue
        if b["sha256"] is None or sha256(b["crop"]) != b["sha256"]:
            raise SystemExit(f"{b['crop']}: checksum does not match integrity.json")
        S = b["size"]
        reused = reuse / "crops" / b["obs"] / f"{b['block_id']}.png" if reuse is not None else None
        if reused is not None and reused.is_file() and not (args.quarter_train and split == "train" and S > 512):
            u8 = np.array(Image.open(reused))
        else:
            u8 = None
        raw = np.array(Image.open(b["crop"])) if u8 is None else np.zeros((b["h"], b["w"]))
        if raw.shape != (b["h"], b["w"]):
            raise SystemExit(f"{b['crop']}: shape {raw.shape} != labels.csv {b['h']}x{b['w']}")
        S = b["size"]
        if raw.shape != (S, S):  # partial edge crop: pad right/bottom with nodata
            raw = np.pad(raw, ((0, S - raw.shape[0]), (0, S - raw.shape[1])))
        if u8 is None:
            u8 = to_uint8(raw, b["export"]["meta"].get("preprocessing", {}), args.stretch)
            if args.blur_sigma > 0:
                u8 = blur_valid(u8, args.blur_sigma)
        base_row = {"label": b["label"], "split": split, "obs_id": b["obs"], "expert_class": b["expert_class"],
                    "edit_count": b["edit_count"], "cell": cell, "fold": "" if fold is None else fold,
                    "cv_buffer": ";".join(map(str, border)), "parent_block_id": b["block_id"]}
        pieces = [(b["block_id"], 0, 0, S)]
        if args.quarter_train and split == "train" and S > 512:
            pieces = [(f"{b['block_id']}_q{r}{c}", r * 512, c * 512, 512)
                      for r in range(S // 512) for c in range(S // 512)]
        for block_id, r0, c0, size in pieces:
            rel = Path("crops") / b["obs"] / f"{block_id}.png"
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ascontiguousarray(u8[r0:r0 + size, c0:c0 + size])).save(out_dir / rel)
            rows.append({"image": rel.as_posix(), "block_id": block_id, "x_px": b["x"] + c0,
                         "y_px": b["y"] + r0, "size": size, **base_row})
        v = u8[u8 > 0][::61].astype(np.float64)
        st = stats.setdefault(b["obs"], [0, 0.0, 0.0])
        st[0] += v.size
        st[1] += v.sum()
        st[2] += (v ** 2).sum()

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = [{"name": c["name"], "seg_index": i} for i, c in enumerate(dc)]
    fixed += [{"name": c["name"], "seg_index": None, "expert_class_id": c["id"]}
              for c in exports[0]["classes"] if int(c["model_index"]) >= num_seg]
    (out_dir / "block_classes.json").write_text(json.dumps({"classes": fixed}, indent=2))
    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["image", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # --- report -------------------------------------------------------------------
    table = Counter((b["expert_class"], s) for b, s in zip(blocks, splits))
    cols = (*SPLITS, "buffer")
    print(f"\nSplit of all {len(blocks)} labelled blocks ({args.cell_px} px cells, buffer {args.buffer}):")
    print(f"  {'class':<45}" + "".join(f"{c:>8}" for c in cols))
    for c in exports[0]["classes"]:
        counts = [table[(c["name"], s)] for s in cols]
        if any(counts):
            print(f"  {c['name']:<45}" + "".join(f"{n:>8}" for n in counts))
    for o in sorted({b["obs"] for b in blocks}):
        n = Counter(s for b, s in zip(blocks, splits) if b["obs"] == o)
        tot = sum(n.values())
        print(f"  {o}: " + "  ".join(f"{s} {n[s]} ({n[s] / tot:.0%})" for s in cols))
    if args.cv_folds:
        fold_table = Counter((b["expert_class"], f) for b, f in zip(blocks, folds) if f is not None)
        print(f"\nCross-validation folds over train+val ({args.cv_folds} folds, whole cells):")
        print(f"  {'class':<45}" + "".join(f"{'fold ' + str(f):>8}" for f in range(args.cv_folds)))
        for c in exports[0]["classes"]:
            counts = [fold_table[(c["name"], f)] for f in range(args.cv_folds)]
            if any(counts):
                print(f"  {c['name']:<45}" + "".join(f"{n:>8}" for n in counts))
    summary = {"block_sizes": args.block_sizes, "keep_splits_from": args.keep_splits_from,
               "quarter_train": args.quarter_train, "cv_folds": args.cv_folds,
               "stretch": args.stretch, "blur_sigma": args.blur_sigma, "oxia_p1_p99": OXIA_P1_P99, "cell_px": args.cell_px,
               "fractions": list(args.fractions), "buffer": args.buffer, "seed": args.seed,
               "blocks_labelled": len(blocks), "blocks_in_manifest": len(rows), "missing_crops": len(missing),
               "exports": [str(e["folder"]) for e in exports],
               "split_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(table.items())},
               "pixel_stats": {}}
    for o, (n, s1, s2) in sorted(stats.items()):
        mean = s1 / n
        std = (s2 / n - mean ** 2) ** 0.5
        summary["pixel_stats"][o] = {"mean": round(mean, 1), "std": round(std, 1)}
        print(f"  {o}: converted valid pixels mean {mean:.1f}, std {std:.1f} (Oxia training: 119.3, 24.9)")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(rows)} blocks to {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
