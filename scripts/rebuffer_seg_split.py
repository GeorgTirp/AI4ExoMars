#!/usr/bin/env python3
"""Re-assign the seg manifest's train/val split with a spatial buffer.

`prep_seg_crops.py` assigns the split by crop CENTRE:

    split = "val" if (col + S // 2) >= val_col_start else "train"

so a train crop whose centre sits just left of the boundary still extends
S//2 = 256 px into the held-out band, `--spatial-jitter-px` pushes that another
32 px, and -- the part that actually matters -- the context branch reads a
2048 px window reaching 1024 px beyond the crop centre. The leak is therefore
LARGER for the context variants (v1/v3) than for the no-context ones (v0/v2),
which biases exactly the comparison the variant sweep exists to make.

This script rewrites ONLY the `split` column. Row order is untouched, so every
`rec.index` keeps its meaning and both index-addressed caches
(seg_crop_cache_full, seg_context_cache_full) stay valid -- no 37 GB rebuild.
Crops whose footprint straddles the boundary are marked `buffer`, which
`partition_records` puts in neither split.

One buffer is applied to ALL variants, sized for the widest footprint any of
them uses (the context window). Using a per-variant buffer would give each
variant a different training set and make the comparison meaningless.

    python scripts/rebuffer_seg_split.py --manifest <csv> --dry-run
    python scripts/rebuffer_seg_split.py --manifest <csv> --half-width 1024
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--half-width", type=int, default=1024,
        help="Half-width of the footprint to keep clear of the boundary, from "
             "the crop CENTRE. 1024 = the context window's reach (the binding "
             "constraint). 288 = local crop (256) + jitter (32) only, which is "
             "enough for v0/v2 but leaves v1/v3 leaking.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.manifest)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    sizes = {int(r["size"]) for r in rows}
    if len(sizes) != 1:
        raise SystemExit(f"mixed crop sizes {sizes}; this assumes one uniform size")
    S = sizes.pop()

    # Recover the original boundary: the smallest crop centre that prep_seg_crops
    # classified as val. Reproduces its assignment exactly.
    val_centres = [int(r["col"]) + S // 2 for r in rows if r["split"] == "val"]
    if not val_centres:
        raise SystemExit("manifest has no val rows -- nothing to re-buffer")
    boundary = min(val_centres)

    H = args.half_width
    counts = {"train": 0, "val": 0, "buffer": 0}
    for row in rows:
        centre = int(row["col"]) + S // 2
        if centre + H < boundary:
            row["split"] = "train"
        elif centre - H >= boundary:
            row["split"] = "val"
        else:
            row["split"] = "buffer"
        counts[row["split"]] += 1

    total = len(rows)
    print(f"manifest        : {path}")
    print(f"crop size       : {S}   boundary col: {boundary}")
    print(f"buffer half-width: {H} px (footprint kept clear of the boundary)")
    print(f"train           : {counts['train']:,}")
    print(f"val             : {counts['val']:,}")
    print(f"buffer (dropped): {counts['buffer']:,}  ({100*counts['buffer']/total:.1f}% of {total:,})")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    backup = path.with_suffix(path.suffix + ".prebuffer")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup          : {backup}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rewrote {total:,} rows (order preserved -- caches remain valid)")


if __name__ == "__main__":
    main()
