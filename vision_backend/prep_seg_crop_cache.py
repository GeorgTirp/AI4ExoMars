#!/usr/bin/env python3
"""Precompute a padded (imagery, label) crop cache for Stage-3 segmentation.

Both `drg_on_label_grid.tif` and `labels_DC_classid.tif` are DEFLATE-tiled at
exactly the crop size (512px). A non-jittered read hits one tile, but
`SegmentationCropDataset`'s runtime spatial jitter shifts the window off that
tile grid, spanning up to 4 tiles that get freshly decompressed on every
single draw -- once per epoch per crop, for the lifetime of every future
training run against a given manifest.

This script reads each manifest window exactly once, padded by
`--jitter-margin` px on every side, and writes the raw uint8 pixels into two
packed memmap arrays (images.npy, labels.npy; shape (N, S+2*margin, S+2*margin)).
`SegmentationCropDataset(cache_dir=...)` then slices the jittered crop straight
out of the cache instead of touching the source GeoTIFFs at all. Label pairing
(ground-bounds windowing across the imagery/label rasters' differing
transforms) exactly mirrors what `__getitem__` does live, just on the padded
window instead of the exact crop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

try:
    from vision_backend.seg_dataset import load_seg_records
except ModuleNotFoundError:
    from seg_dataset import load_seg_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="seg-crop manifest CSV (prep_seg_crops output)")
    ap.add_argument("--imagery", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--jitter-margin", type=int, default=32,
                    help="Pad every crop by this many px on each side, so a training run's "
                         "--spatial-jitter-px (must be <= this) can slice straight from the cache.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    records = load_seg_records(args.manifest)
    if not records:
        raise SystemExit(f"No records in {args.manifest!r}")

    sizes = {rec.size for rec in records}
    if len(sizes) != 1:
        raise SystemExit(f"Manifest has mixed crop sizes {sizes}; cache assumes one uniform size.")
    S = sizes.pop()
    J = args.jitter_margin
    P = S + 2 * J
    n = len(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(n, P, P))
    labels = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+", dtype=np.uint8, shape=(n, P, P))

    est_gb = n * P * P * 2 / 1e9
    print(f"{n} crops, padded {P}x{P} (crop {S} + margin {J}) -> ~{est_gb:.1f} GB in {out_dir}")

    with rasterio.open(args.imagery) as img_ds, rasterio.open(args.labels) as lab_ds:
        for i, rec in enumerate(records):
            assert rec.index == i, "load_seg_records ordering must match cache row order"
            win = Window(rec.col - J, rec.row - J, P, P)
            images[i] = img_ds.read(1, window=win, boundless=True, fill_value=0)

            bounds = rasterio.windows.bounds(win, img_ds.transform)
            lab_win = from_bounds(*bounds, transform=lab_ds.transform).round_offsets().round_lengths()
            labels[i] = lab_ds.read(1, window=lab_win, out_shape=(P, P), boundless=True, fill_value=0)

            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{n}", flush=True)

    images.flush()
    labels.flush()

    meta = {
        "crop_size": S,
        "jitter_margin": J,
        "padded_size": P,
        "count": n,
        "manifest": str(args.manifest),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n} crops -> {out_dir}")


if __name__ == "__main__":
    main()
