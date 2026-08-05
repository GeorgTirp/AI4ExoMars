"""Step 3 -- warp the DRG mosaic onto the NOAH-H label grid (Route A).

The DRG is geographic lon/lat on the Mars 2000 *sphere* (+proj=longlat
+R=3396190); the labels are projected eqc metres on the Mars 2000 *ellipsoid*
(rf=169.894), lon_0=0. Both CRSs are fully defined, so PROJ does an explicit
transform -- no null datum shift. The output grid is snapped exactly to the
label grid (same CRS, pixel size 0.241384614 m, and origin phase), so decoded
labels overlay the warped imagery pixel-for-pixel.

Output is a target-grid-aligned window (an AOI), not the whole scene, so it is
cheap to produce and verify. The full-scene command is identical with a larger
--aoi (see README).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from common import DERIVED_DIR, DRG_VRT, read_target_grid

# DRG mosaic overlap centre (deg) -- default AOI sits here, well inside labels.
DRG_CENTER_LON = -24.27695
DRG_CENTER_LAT = 18.20703
A_MARS = 3396190.0
DEG2M = A_MARS * math.pi / 180.0  # eqc metres per degree at lat_ts=0


def lonlat_to_target_xy(lon: float, lat: float) -> tuple[float, float]:
    """eqc forward on the label ellipsoid (lat_ts=0): x=a*lon, y=a*lat (rad)."""
    return lon * DEG2M, lat * DEG2M


def snap_window_to_grid(grid, xmin, ymin, xmax, ymax) -> Window:
    """Integer pixel window in the label grid covering the AOI bbox."""
    col0 = int(math.floor((xmin - grid.left) / grid.res_x))
    row0 = int(math.floor((grid.top - ymax) / grid.res_y))
    col1 = int(math.ceil((xmax - grid.left) / grid.res_x))
    row1 = int(math.ceil((grid.top - ymin) / grid.res_y))
    return Window(col0, row0, col1 - col0, row1 - row0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DRG_VRT), help="DRG VRT or single tile")
    ap.add_argument("--aoi", nargs=4, type=float, default=None,
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="AOI in TARGET CRS metres; default = centred box of --size")
    ap.add_argument("--center-lonlat", nargs=2, type=float,
                    default=(DRG_CENTER_LON, DRG_CENTER_LAT),
                    help="AOI centre in lon/lat when --aoi is not given")
    ap.add_argument("--size", type=int, default=8192,
                    help="AOI side length in label pixels (square) for centred mode")
    ap.add_argument("--resampling", default="cubic",
                    choices=[r.name for r in Resampling if r.value <= 7])
    ap.add_argument("--block", type=int, default=4096,
                    help="output block size for streaming warp (memory-bounded)")
    ap.add_argument("--out", default=str(DERIVED_DIR / "drg_on_label_grid_aoi.tif"))
    args = ap.parse_args()

    grid = read_target_grid()
    label_crs = rasterio.crs.CRS.from_wkt(grid.crs_wkt)

    if args.aoi:
        xmin, ymin, xmax, ymax = args.aoi
    else:
        cx, cy = lonlat_to_target_xy(*args.center_lonlat)
        half = args.size * grid.res_x / 2.0
        xmin, ymin, xmax, ymax = cx - half, cy - half, cx + half, cy + half

    win = snap_window_to_grid(grid, xmin, ymin, xmax, ymax)
    dst_transform = window_transform(win, grid.affine)
    width, height = int(win.width), int(win.height)

    print(f"AOI (target CRS m): x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}]")
    print(f"snapped window    : col{win.col_off} row{win.row_off} {width}x{height}")
    print(f"dst origin (TL)   : x={dst_transform.c:.4f} y={dst_transform.f:.4f}")

    profile = dict(
        driver="GTiff", width=width, height=height, count=1, dtype="uint8",
        nodata=0, crs=label_crs, transform=dst_transform,
        tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", predictor=2, zlevel=6, BIGTIFF="YES",
    )

    # Stream the warp block by block so a full-scene (tens of gigapixel) output
    # never materialises in RAM. Each output window is warped on demand by the
    # WarpedVRT (GDAL reads only the source tiles it needs).
    B = args.block
    total_px = width * height
    valid = 0
    t0 = time.time()
    n_blocks = ((height + B - 1) // B) * ((width + B - 1) // B)
    done = 0

    with rasterio.open(args.src) as src:
        print(f"src CRS           : {src.crs.to_proj4()}")
        print(f"streaming warp    : {n_blocks} blocks of {B}px, resampling="
              f"{args.resampling}")
        with WarpedVRT(
            src, crs=label_crs, transform=dst_transform, width=width,
            height=height, resampling=Resampling[args.resampling],
            src_nodata=0, nodata=0,
        ) as vrt, rasterio.open(args.out, "w", **profile) as dst:
            for row in range(0, height, B):
                h = min(B, height - row)
                for col in range(0, width, B):
                    w = min(B, width - col)
                    win = Window(col, row, w, h)
                    data = vrt.read(1, window=win)
                    dst.write(data, 1, window=win)
                    valid += int((data > 0).sum())
                    done += 1
                    if done % 20 == 0 or done == n_blocks:
                        dt = time.time() - t0
                        print(f"  {done}/{n_blocks} blocks  "
                              f"({100*done/n_blocks:.0f}%)  {dt:.0f}s  "
                              f"valid so far {valid:,}", flush=True)

    print(f"warped            : {width}x{height}, valid {valid}/{total_px} "
          f"({100*valid/total_px:.1f}%) in {time.time()-t0:.0f}s")
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    main()
