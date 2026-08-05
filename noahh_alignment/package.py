"""Step 6 -- add internal overviews to a target-grid raster (COG-style).

The GDAL CLI (gdaladdo) is not installed, so overviews are built through
rasterio. The warp/decode outputs are already tiled + DEFLATE; adding internal
overviews makes them fast for windowed/zoomed reads by downstream tools.
"""

from __future__ import annotations

import argparse

import rasterio
from rasterio.enums import Resampling

from common import DERIVED_DIR


def add_overviews(path: str, *, resampling: str, factors=(2, 4, 8, 16, 32)) -> None:
    with rasterio.open(path, "r+") as ds:
        ds.build_overviews(list(factors), Resampling[resampling])
        ds.update_tags(ns="rio_overview", resampling=resampling)
    with rasterio.open(path) as ds:
        print(f"{path}\n  overviews(b1): {ds.overviews(1)}  ({ds.width}x{ds.height})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="rasters to add overviews to")
    ap.add_argument("--imagery", action="store_true",
                    help="use average resampling (default nearest for class ids)")
    args = ap.parse_args()

    targets = args.targets or [
        str(DERIVED_DIR / "drg_on_label_grid_aoi.tif"),
        str(DERIVED_DIR / "labels_DC_classid.tif"),
    ]
    for t in targets:
        # imagery -> average; class-index rasters -> nearest (never blend ids)
        res = "average" if (args.imagery or "drg_on_label" in t) else "nearest"
        add_overviews(t, resampling=res)


if __name__ == "__main__":
    main()
