#!/usr/bin/env python3
"""Precompute the CONTEXT crop cache for the context-aware Stage-3 encoder.

`ContextAwareConvNeXtSwinEncoder`'s second branch wants a wide, downsampled view
of the ground surrounding each local crop -- a `--context-size` window (2048px by
default, i.e. 8x the 512px crop) centred on the crop and resized to
`--context-output-size`. The labelled segmentation loader has never produced one:
only the Stage-1 SSL loader does. Context-on Stage-3 variants therefore need this
cache before they can train at all.

Row i here corresponds to record i of the same manifest `prep_seg_crop_cache.py`
consumed, so `context.npy` is INDEX-ALIGNED with that script's `images.npy` /
`labels.npy` and the dataset can slice both by `rec.index`. That alignment is the
whole correctness property: a context crop attached to the wrong local crop would
train the model on a mismatched neighbourhood and be invisible in the loss.

Windows are read boundless (like `SegmentationCropDataset._read_live`), so a
context window spilling past the swath edge comes back 0 = nodata, exactly as
real off-swath imagery would.

Storage is uint8 at `--context-output-size` (default 512), i.e. the same bytes
per row as one local crop -- the 2048px window is downsampled on write, never
stored at full size.

Run this ON THE SERVER; it reads the full mosaic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

try:
    from vision_backend.seg_dataset import load_seg_records
except ModuleNotFoundError:
    from seg_dataset import load_seg_records


def context_window(rec, context_size: int) -> Window:
    """The `context_size` window centred on this record's crop.

    Centring (rather than anchoring at the crop's origin) is what makes the
    context a *surrounding* view: the local crop sits in the middle of it.
    """
    half = context_size // 2
    centre_col = rec.col + rec.size // 2
    centre_row = rec.row + rec.size // 2
    return Window(centre_col - half, centre_row - half, context_size, context_size)


def verify_cache(out_dir: Path, manifest: str, imagery: str, samples: int,
                 seed: int = 0) -> int:
    """Re-read random rows live and compare against the cache, byte for byte.

    A row-count check cannot catch a REORDERED cache -- the failure that pairs
    every crop with someone else's surroundings and still trains happily. This
    re-derives the window for a random sample of records and compares content,
    which does catch it. Returns the number of mismatches.
    """
    import random

    context = np.load(out_dir / "context.npy", mmap_mode="r")
    meta = json.loads((out_dir / "context_meta.json").read_text())
    records = load_seg_records(manifest)

    if context.shape[0] != len(records):
        print(f"FAIL: cache has {context.shape[0]} rows, manifest has {len(records)}")
        return context.shape[0]

    C, O = int(meta["context_size"]), int(meta["context_output_size"])
    picks = random.Random(seed).sample(range(len(records)), min(samples, len(records)))
    mismatches = 0
    with rasterio.open(imagery) as img_ds:
        for i in picks:
            rec = records[i]
            live = img_ds.read(
                1, window=context_window(rec, C), out_shape=(O, O),
                boundless=True, fill_value=0, resampling=Resampling.average,
            )
            if not np.array_equal(np.asarray(context[rec.index]), live):
                mismatches += 1
                print(f"  MISMATCH at row {rec.index} (col={rec.col}, row={rec.row})")

    if mismatches:
        print(f"FAIL: {mismatches}/{len(picks)} sampled rows do not match a live read. "
              f"The cache is misaligned -- do NOT train on it; rebuild it.")
    else:
        print(f"OK: {len(picks)} randomly sampled rows match a live read exactly "
              f"(content-checked, not just counted).")
    return mismatches


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True,
                    help="seg-crop manifest CSV -- MUST be the same one used for "
                         "prep_seg_crop_cache.py, or the caches won't align.")
    ap.add_argument("--imagery", required=True,
                    help="Imagery raster (drg_on_label_grid.tif), same as the local cache.")
    ap.add_argument("--context-size", type=int, default=2048,
                    help="Side length in NATIVE px of the window centred on each crop.")
    ap.add_argument("--context-output-size", type=int, default=512,
                    help="Side length the context window is downsampled to for storage "
                         "and for the model input.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verify-samples", type=int, default=32,
                    help="After building, re-read this many random rows live and "
                         "compare content. 0 disables.")
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip the build; only content-check an existing cache.")
    args = ap.parse_args()

    if args.verify_only:
        failures = verify_cache(Path(args.out_dir), args.manifest, args.imagery,
                                max(1, args.verify_samples))
        raise SystemExit(1 if failures else 0)

    records = load_seg_records(args.manifest)
    if not records:
        raise SystemExit(f"No records in {args.manifest!r}")

    C = int(args.context_size)
    O = int(args.context_output_size)
    if C <= 0 or O <= 0:
        raise SystemExit("--context-size and --context-output-size must be positive")
    n = len(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = np.lib.format.open_memmap(
        out_dir / "context.npy", mode="w+", dtype=np.uint8, shape=(n, O, O)
    )

    est_gb = n * O * O / 1e9
    print(f"{n} crops, context {C}x{C} native -> stored {O}x{O} uint8 "
          f"(~{est_gb:.1f} GB) in {out_dir}")

    with rasterio.open(args.imagery) as img_ds:
        for i, rec in enumerate(records):
            # load_seg_records assigns rec.index in file order; the local cache
            # relies on the same invariant, so this is what keeps them aligned.
            assert rec.index == i, "load_seg_records ordering must match cache row order"
            win = context_window(rec, C)
            context[i] = img_ds.read(
                1, window=win, out_shape=(O, O),
                boundless=True, fill_value=0,
                resampling=Resampling.average,
            )
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{n}", flush=True)

    context.flush()

    meta = {
        "context_size": C,
        "context_output_size": O,
        "count": n,
        "manifest": str(args.manifest),
        "imagery": str(args.imagery),
    }
    (out_dir / "context_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n} context crops -> {out_dir}")

    if args.verify_samples:
        if verify_cache(out_dir, args.manifest, args.imagery, args.verify_samples):
            raise SystemExit("context cache failed its own verification")


if __name__ == "__main__":
    main()
