#!/usr/bin/env python3
"""Stage-1 crop preparation: HiRISE JP2 -> normalized 1024² uint8 .npy crops.

Single-branch SimMIM pretraining needs clean, native-resolution crops with the
nuisance radiometry removed at prep time. This CLI:

  * tiles each observation into non-overlapping 1024² crops (no padding -- the
    remainder strip at the right/bottom edge is dropped);
  * rejects crops with >1% invalid (0/255) pixels or near-zero variance;
  * per-crop normalizes (clip to [p1, p99] -> [-1, 1]) and stores as uint8;
  * keeps observation_id / x / y / gsd per crop in a CSV manifest;
  * writes splits.json partitioned BY OBSERVATION (never by crop).

Reuses readers + the invalid-pixel filter from ``hirise_patchloader``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

try:
    from vision_backend.hirise_patchloader import (
        ArrayImageReader,
        BaseImageReader,
        RasterioImageReader,
        compute_invalid_mask,
        find_jp2_files,
        open_image_reader,
        require_numpy,
        resolve_path,
    )
except ModuleNotFoundError:
    from hirise_patchloader import (  # type: ignore
        ArrayImageReader,
        BaseImageReader,
        RasterioImageReader,
        compute_invalid_mask,
        find_jp2_files,
        open_image_reader,
        require_numpy,
        resolve_path,
    )

CROP_SIZE = 1024
INDEX_FILENAME = "crops_index.csv"
SPLITS_FILENAME = "splits.json"
MANIFEST_FILENAME = "manifest.csv"


@dataclass(frozen=True)
class CropRecord:
    crop_path: str
    source_image: str
    observation_id: str
    row: int
    col: int
    x: int  # left
    y: int  # top
    gsd: float
    crop_size: int
    invalid_fraction: float
    std: float


def manifest_fieldnames() -> list[str]:
    return list(CropRecord.__annotations__.keys())


def iter_full_tiles(height: int, width: int, size: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(row, col, top, left)`` for every fully-contained ``size`` tile."""
    for row, top in enumerate(range(0, height - size + 1, size)):
        for col, left in enumerate(range(0, width - size + 1, size)):
            yield row, col, top, left


def to_2d(arr):
    np = require_numpy()
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 1:
        return a[:, :, 0]
    return a


def normalize_crop(arr, invalid_mask):
    """Per-crop robust stretch -> uint8 in [0, 255] plus the post-norm std.

    Percentiles are computed over valid (non-0/255, finite) pixels. Returns
    ``(uint8_array, std)`` or ``(None, 0.0)`` when the crop is constant.
    The stored uint8 dequantizes to [-1, 1] via ``u8 / 127.5 - 1.0``.
    """
    np = require_numpy()
    a = to_2d(arr).astype(np.float32)

    valid = a[~to_2d(invalid_mask)] if invalid_mask is not None else a.reshape(-1)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None, 0.0

    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, 99.0))
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return None, 0.0

    unit = np.clip((a - lo) / (hi - lo), 0.0, 1.0)  # [0, 1]
    norm = unit * 2.0 - 1.0  # [-1, 1]
    std = float(norm.std())
    u8 = np.rint(unit * 255.0).clip(0, 255).astype(np.uint8)
    return u8, std


def detect_gsd(reader: BaseImageReader) -> Optional[float]:
    """Best-effort ground-sample-distance (m/px) from the rasterio transform."""
    if isinstance(reader, RasterioImageReader):
        try:
            xres, yres = reader._dataset.res
            gsd = abs(float(xres))
            # a degenerate identity transform (1,1) means no georeferencing
            if gsd > 0 and abs(gsd - 1.0) > 1e-9:
                return gsd
        except Exception:
            return None
    return None


def gsd_matches(gsd: Optional[float], target: Optional[float], tol: float) -> bool:
    if target is None:
        return True  # no GSD filtering requested
    if gsd is None:
        return False  # filtering requested but GSD unknown -> exclude
    return abs(gsd - target) <= tol


def process_observation(
    reader: BaseImageReader,
    observation_id: str,
    *,
    source_rel: str,
    output_dir: Path,
    crop_size: int,
    max_invalid_fraction: float,
    min_std: float,
    gsd: float,
    dry_run: bool,
    overwrite: bool,
) -> tuple[list[CropRecord], dict[str, int]]:
    np = require_numpy()
    obs_dir = output_dir / observation_id
    if not dry_run:
        obs_dir.mkdir(parents=True, exist_ok=True)

    records: list[CropRecord] = []
    stats = {"candidates": 0, "kept": 0, "rejected_invalid": 0, "rejected_lowvar": 0}

    for row, col, top, left in iter_full_tiles(reader.height, reader.width, crop_size):
        stats["candidates"] += 1
        crop = reader.read_region(top, left, crop_size, crop_size)
        invalid_mask = compute_invalid_mask(crop)
        invalid_fraction = float(to_2d(invalid_mask).mean())
        if max_invalid_fraction >= 0.0 and invalid_fraction > max_invalid_fraction:
            stats["rejected_invalid"] += 1
            continue

        u8, std = normalize_crop(crop, invalid_mask)
        if u8 is None or std < min_std:
            stats["rejected_lowvar"] += 1
            continue

        crop_name = f"{observation_id}_r{row:04d}_c{col:04d}_y{top:07d}_x{left:07d}.npy"
        crop_path = obs_dir / crop_name
        if not dry_run and (overwrite or not crop_path.exists()):
            np.save(crop_path, u8)

        records.append(
            CropRecord(
                crop_path=os.path.relpath(crop_path, output_dir),
                source_image=source_rel,
                observation_id=observation_id,
                row=row,
                col=col,
                x=left,
                y=top,
                gsd=gsd,
                crop_size=crop_size,
                invalid_fraction=invalid_fraction,
                std=std,
            )
        )
        stats["kept"] += 1

    return records, stats


def split_by_observation(
    observation_ids: list[str],
    *,
    val_fraction: float,
    min_val: int,
    seed: int,
) -> dict[str, list[str]]:
    """Partition observations (not crops) into train/val with a fixed seed."""
    unique = sorted(set(observation_ids))
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)

    n_val = max(min_val, int(round(len(unique) * val_fraction)))
    n_val = min(n_val, len(unique))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return {"train": train, "val": val, "seed": seed, "val_fraction": val_fraction}


def write_manifest(path: Path, records: list[CropRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames())
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def process_one(task: dict) -> dict:
    """Process a single strip (one ProcessPool task): tile, filter, normalize,
    write per-observation manifest + .npy crops. Returns picklable stats.
    """
    image_path = Path(task["image_path"])
    output_dir = Path(task["output_dir"])
    observation_id = image_path.stem
    base = {
        "observation_id": observation_id,
        "gsd_key": "unknown",
        "excluded": False,
        "error": None,
        "candidates": 0,
        "kept": 0,
        "rejected_invalid": 0,
        "rejected_lowvar": 0,
    }
    try:
        with open_image_reader(image_path) as reader:
            gsd = detect_gsd(reader)
            base["gsd_key"] = "unknown" if gsd is None else f"{gsd:.3f}"
            if not gsd_matches(gsd, task["gsd"], task["gsd_tolerance"]):
                base["excluded"] = True
                return base
            records, stats = process_observation(
                reader,
                observation_id,
                source_rel=os.path.relpath(image_path, output_dir)
                if not task["dry_run"]
                else str(image_path),
                output_dir=output_dir,
                crop_size=task["crop_size"],
                max_invalid_fraction=task["max_invalid_fraction"],
                min_std=task["min_std"],
                gsd=gsd if gsd is not None else -1.0,
                dry_run=task["dry_run"],
                overwrite=task["overwrite"],
            )
        if records and not task["dry_run"]:
            write_manifest(output_dir / observation_id / MANIFEST_FILENAME, records)
        base.update(stats)
    except Exception as exc:  # keep the batch going if one strip fails
        base["error"] = str(exc)
    return base


def rebuild_global_index(
    output_dir: Path,
    *,
    val_fraction: float,
    min_val: int,
    seed: int,
) -> tuple[int, int, dict]:
    """Rebuild crops_index.csv + splits.json from ALL per-observation manifests
    present in ``output_dir`` (so incremental batches accumulate)."""
    manifests = sorted(output_dir.glob(f"*/{MANIFEST_FILENAME}"))
    observation_ids: list[str] = []
    total = 0
    with (output_dir / INDEX_FILENAME).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=manifest_fieldnames())
        writer.writeheader()
        for manifest_path in manifests:
            observation_ids.append(manifest_path.parent.name)
            with manifest_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    writer.writerow(row)
                    total += 1
    splits = split_by_observation(
        observation_ids, val_fraction=val_fraction, min_val=min_val, seed=seed
    )
    with (output_dir / SPLITS_FILENAME).open("w") as f:
        json.dump(splits, f, indent=2)
    return total, len(observation_ids), splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True, help="Directory of source rasters (.jp2 / .tif)."
    )
    parser.add_argument("--output-dir", required=True, help="Destination for crops + manifest.")
    parser.add_argument("--glob", default="*.jp2", help="Filename glob, e.g. '*.tif' for NOAH DRG tiles.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument(
        "--max-invalid-fraction",
        type=float,
        default=0.01,
        help="Drop crops with more than this fraction of 0/255 pixels.",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=0.02,
        help="Drop crops whose post-normalization std (on [-1,1]) is below this.",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Single-GSD policy: keep only observations at this m/px (e.g. 0.25 or 0.5). "
        "Omit to keep all GSDs.",
    )
    parser.add_argument("--gsd-tolerance", type=float, default=0.02)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-val-observations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index into the sorted strip list to start at (for batching).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many strips from --start-index (batch size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Strips to process in parallel (one process per strip).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    all_images = find_jp2_files(input_dir, args.glob, args.recursive)
    if not all_images:
        print(
            f"No raster images found in {input_dir} (glob {args.glob!r}). "
            f"Recognized suffixes: .jp2 .tif .tiff .vrt; pass --recursive if the "
            f"rasters sit in subdirectories.",
            file=sys.stderr,
        )
        return 1

    start = max(0, args.start_index)
    images = all_images[start:]
    if args.max_images is not None and args.max_images >= 0:
        images = images[: args.max_images]
    if not images:
        print(
            f"No strips selected (have {len(all_images)}, --start-index {start}).",
            file=sys.stderr,
        )
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    end = start + len(images)
    print(
        f"[prep] strips {start}..{end - 1} of {len(all_images)} "
        f"({len(images)} this batch); workers={args.num_workers}; GSD policy: "
        f"{'all' if args.gsd is None else f'{args.gsd} +/- {args.gsd_tolerance} m/px'}"
    )

    tasks = [
        {
            "image_path": str(image_path),
            "output_dir": str(output_dir),
            "crop_size": args.crop_size,
            "max_invalid_fraction": args.max_invalid_fraction,
            "min_std": args.min_std,
            "gsd": args.gsd,
            "gsd_tolerance": args.gsd_tolerance,
            "dry_run": args.dry_run,
            "overwrite": args.overwrite,
        }
        for image_path in images
    ]

    def report(r: dict) -> None:
        if r["error"]:
            print(f"  - {r['observation_id']}: ERROR {r['error']}")
        elif r["excluded"]:
            print(f"  - {r['observation_id']}: excluded by GSD policy (gsd={r['gsd_key']})")
        else:
            print(
                f"  - {r['observation_id']}: {r['kept']}/{r['candidates']} kept "
                f"(invalid {r['rejected_invalid']}, low-var {r['rejected_lowvar']}), gsd={r['gsd_key']}"
            )

    results: list[dict] = []
    if args.num_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for r in executor.map(process_one, tasks):
                results.append(r)
                report(r)
    else:
        for task in tasks:
            r = process_one(task)
            results.append(r)
            report(r)

    gsd_counts: dict[str, int] = {}
    for r in results:
        gsd_counts[r["gsd_key"]] = gsd_counts.get(r["gsd_key"], 0) + 1
    excluded = sum(1 for r in results if r["excluded"])
    errors = sum(1 for r in results if r["error"])
    batch_kept = sum(r["kept"] for r in results)
    print(f"[prep] GSD distribution: {gsd_counts}; excluded by policy: {excluded}; errors: {errors}")
    print(f"[prep] This batch: {batch_kept} crops across {len(results)} strip(s).")

    if args.dry_run:
        return 0

    total, n_obs, splits = rebuild_global_index(
        output_dir,
        val_fraction=args.val_fraction,
        min_val=args.min_val_observations,
        seed=args.seed,
    )
    print(
        f"[prep] Rebuilt {INDEX_FILENAME} + {SPLITS_FILENAME} (cumulative): "
        f"{total} crops across {n_obs} observation(s); "
        f"train {len(splits['train'])} / val {len(splits['val'])}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
