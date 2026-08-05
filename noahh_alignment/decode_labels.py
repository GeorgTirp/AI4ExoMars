"""Step 7 -- decode an RGBA NOAH-H label mosaic to a single-band class-index
GeoTIFF on the identical grid (no resampling).

  * nearest-colour match each opaque pixel to the legend (tolerance handles the
    rendered 0->3 channel shift); 1-based class ids.
  * alpha == 0 -> nodata (0).  '#000000' stays the Boulder-fields class.
  * attaches a colormap and writes classes.json.

Runs full-frame or over an AOI window (for verification / partial extraction).
Usage:
  python decode_labels.py --kind DC                       # full mosaic
  python decode_labels.py --kind DC --window C R W H       # AOI (pixel coords)
  python decode_labels.py --kind DC --window C R W H --stats-only   # no write
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rasterio
from rasterio.windows import Window

from common import (
    DC_LABEL,
    DC_LEGEND,
    DERIVED_DIR,
    IG_LABEL,
    IG_LEGEND,
    build_color_lut,
    build_palette,
    decode_block_with_lut,
    gdal_colormap,
    parse_legend,
    read_target_grid,
    write_classes_json,
)

LABELS = {"DC": (DC_LABEL, DC_LEGEND), "IG": (IG_LABEL, IG_LEGEND)}


def iter_blocks(col0, row0, width, height, bw, bh):
    for r in range(row0, row0 + height, bh):
        h = min(bh, row0 + height - r)
        for c in range(col0, col0 + width, bw):
            w = min(bw, col0 + width - c)
            yield c, r, w, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(LABELS), default="DC")
    ap.add_argument("--window", nargs=4, type=int, metavar=("COL", "ROW", "W", "H"),
                    help="AOI in source pixel coords; default = whole mosaic")
    ap.add_argument("--block", nargs=2, type=int, default=(8192, 512),
                    metavar=("BW", "BH"), help="processing block size")
    ap.add_argument("--tol-sq", type=float, default=1600.0,
                    help="max squared-L2 colour distance to accept a class; "
                         "1600 = quarter of the 6400 min inter-class distance, "
                         "so matches are unambiguous and only midpoint-blend "
                         "boundary pixels fall through to nodata")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats-only", action="store_true",
                    help="decode + report histogram but do not write a raster")
    args = ap.parse_args()

    label_path, legend_path = LABELS[args.kind]
    if not label_path.exists():
        raise SystemExit(
            f"[blocked] {args.kind} label mosaic is missing: {label_path.name}\n"
            f"          (only its .aux.xml sidecar is present locally)."
        )

    entries = parse_legend(legend_path)
    palette = build_palette(entries)
    lut = build_color_lut(palette, tol_sq=args.tol_sq)
    grid = read_target_grid(label_path)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    classes_json = DERIVED_DIR / f"classes_{args.kind}.json"
    write_classes_json(entries, classes_json, kind=args.kind)

    with rasterio.open(label_path) as src:
        if args.window:
            col0, row0, width, height = args.window
        else:
            col0, row0, width, height = 0, 0, src.width, src.height

        bw, bh = args.block
        hist = np.zeros(len(entries) + 1, dtype=np.int64)  # index 0 == nodata
        alpha0_px = 0  # genuine nodata (transparent)
        uncertain_px = 0  # opaque but beyond colour tolerance (blend/junk)
        t0 = time.time()

        dst = None
        if not args.stats_only:
            suffix = "_aoi" if args.window else ""
            out_path = args.out or str(
                DERIVED_DIR / f"labels_{args.kind}_classid{suffix}.tif"
            )
            win_transform = src.window_transform(Window(col0, row0, width, height))
            profile = dict(
                driver="GTiff", width=width, height=height, count=1,
                dtype="uint8", nodata=0, crs=src.crs, transform=win_transform,
                tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", predictor=1, zlevel=6, BIGTIFF="YES",
            )
            dst = rasterio.open(out_path, "w", **profile)

        n_blocks = 0
        for c, r, w, h in iter_blocks(col0, row0, width, height, bw, bh):
            block = src.read([1, 2, 3, 4], window=Window(c, r, w, h))
            rgb = np.transpose(block[:3], (1, 2, 0))  # (h, w, 3)
            alpha = block[3]
            cid = decode_block_with_lut(rgb, alpha, lut)
            hist += np.bincount(cid.ravel(), minlength=len(entries) + 1)
            a0 = alpha == 0
            alpha0_px += int(a0.sum())
            uncertain_px += int(((cid == 0) & ~a0).sum())
            if dst is not None:
                dst.write(cid, 1, window=Window(c - col0, r - row0, w, h))
            n_blocks += 1

        if dst is not None:
            dst.write_colormap(1, gdal_colormap(entries))
            dst.close()

    dt = time.time() - t0
    total = int(hist.sum())
    opaque = int(hist[1:].sum())
    print(f"=== decode {args.kind}  ({width}x{height} px, {n_blocks} blocks, {dt:.1f}s) ===")
    print(f"  transparent (alpha=0) nodata : {alpha0_px:,}  ({100*alpha0_px/total:.2f}%)")
    print(f"  opaque, beyond colour tol    : {uncertain_px:,}  "
          f"({100*uncertain_px/total:.2f}%)  [boundary blends -> nodata]")
    print(f"  class-labelled pixels        : {opaque:,}  ({100*opaque/total:.2f}%)")
    denom_op = max(opaque + uncertain_px, 1)
    print(f"  labelled / opaque            : {100*opaque/denom_op:.2f}%")
    for e in entries:
        n = int(hist[e.class_id])
        if n:
            print(f"    {e.class_id:2d} {e.name:52s} {n:>14,}  ({100*n/total:6.3f}%)")
    if not args.stats_only:
        print(f"  -> wrote {out_path}")
    print(f"  -> classes json: {classes_json}")


if __name__ == "__main__":
    main()
