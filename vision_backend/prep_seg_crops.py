"""Build a paired-crop manifest for Stage-3 segmentation.

Scans the co-registered imagery + label rasters (both on the NOAH-H label grid),
keeps tiles that are sufficiently valid (imagery) and labelled (classes), and
writes a manifest CSV (col,row,size in imagery pixels + train/val split).

Tile coverage is evaluated on a *decimated* in-memory read (via the rasters'
overviews) so the scan does not touch every full-resolution pixel. Windows are
emitted in full-resolution coordinates for the dataset to read.

Split: a contiguous right-hand column band of width ``--val-fraction`` is held
out for validation (spatial holdout -> no train/val leakage across adjacent
tiles).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagery", required=True, help="DRG warped onto the label grid")
    ap.add_argument("--labels", required=True, help="class-index raster on the same grid")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None, help="default = crop-size")
    ap.add_argument("--decim", type=int, default=8, help="decimation for the coverage scan")
    ap.add_argument("--min-valid-frac", type=float, default=0.99,
                    help="min fraction of imagery pixels that are valid (>0)")
    ap.add_argument("--min-label-frac", type=float, default=0.50,
                    help="min fraction of pixels carrying a class (>0)")
    ap.add_argument("--val-fraction", type=float, default=0.15,
                    help="right-hand column band held out for validation")
    ap.add_argument("--out", required=True, help="manifest CSV path")
    args = ap.parse_args()

    S = args.crop_size
    stride = args.stride or S
    D = args.decim
    s = max(S // D, 1)

    with rasterio.open(args.imagery) as img, rasterio.open(args.labels) as lab:
        W, H = img.width, img.height
        hs, ws = H // D, W // D
        print(f"imagery {W}x{H}; scanning decimated {ws}x{hs} (decim {D})")
        img_small = img.read(1, out_shape=(hs, ws))
        # labels over the SAME ground extent, resampled to the same decimated grid
        full = Window(0, 0, W, H)
        bounds = rasterio.windows.bounds(full, img.transform)
        lab_win = from_bounds(*bounds, transform=lab.transform)
        lab_small = lab.read(1, window=lab_win, out_shape=(hs, ws),
                             boundless=True, fill_value=0)

        val_col_start = int(round(W * (1.0 - args.val_fraction)))
        rows = list(range(0, H - S + 1, stride))
        cols = list(range(0, W - S + 1, stride))

        kept = 0
        n_train = n_val = 0
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[
                "col", "row", "size", "split", "valid_frac", "label_frac"])
            wr.writeheader()
            for row in rows:
                r0 = row // D
                for col in cols:
                    c0 = col // D
                    im = img_small[r0:r0 + s, c0:c0 + s]
                    lb = lab_small[r0:r0 + s, c0:c0 + s]
                    if im.size == 0:
                        continue
                    vfrac = float((im > 0).mean())
                    lfrac = float((lb > 0).mean())
                    if vfrac < args.min_valid_frac or lfrac < args.min_label_frac:
                        continue
                    split = "val" if (col + S // 2) >= val_col_start else "train"
                    wr.writerow({
                        "col": col, "row": row, "size": S, "split": split,
                        "valid_frac": round(vfrac, 4), "label_frac": round(lfrac, 4),
                    })
                    kept += 1
                    n_train += split == "train"
                    n_val += split == "val"

    print(f"kept {kept} tiles of {len(rows)*len(cols)} candidates "
          f"(train {n_train}, val {n_val})")
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    main()
