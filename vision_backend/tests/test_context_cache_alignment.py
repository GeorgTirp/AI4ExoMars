"""The context cache must be index-aligned with the local crop cache.

This is the highest-risk silent failure in the context-on variants. `context.npy`
rows are addressed by `SegCropRecord.index`, the same index that addresses
`images.npy`. If the two ever disagree in ORDER, every sample pairs a crop with
some other crop's surroundings -- and nothing crashes. Shapes match, counts
match, training runs, the loss goes down, and the resulting numbers are
plausible-but-meaningless.

A row-count assertion cannot catch that, so these tests are content-based: the
fixture raster encodes each tile's position in its pixel VALUES, which makes a
mis-pairing detectable. The reordering test then proves the check has teeth by
deliberately permuting a correct cache and requiring the check to fail.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from vision_backend.prep_seg_context_cache import context_window
from vision_backend.seg_dataset import (
    SegmentationCropDataset,
    load_seg_records,
    partition_records,
)

IMG = 4096
CROP = 512
TILE = 512  # value-encoding granularity


def _encode(row: int, col: int) -> int:
    """Value stamped on tile (row, col). Distinct per tile, fits in uint8."""
    return (row // TILE) * 16 + (col // TILE) * 1 + 1


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A raster whose pixel values encode position + a manifest covering it."""
    tmp = tmp_path_factory.mktemp("ctx")
    rows = np.arange(IMG)[:, None]
    cols = np.arange(IMG)[None, :]
    img = ((rows // TILE) * 16 + (cols // TILE)).astype(np.uint8) + 1
    lab = np.full((IMG, IMG), 3, np.uint8)
    for name, arr in (("img.tif", img), ("lab.tif", lab)):
        with rasterio.open(
            tmp / name, "w", driver="GTiff", height=IMG, width=IMG, count=1,
            dtype=np.uint8, crs="EPSG:4326", transform=Affine.identity(),
        ) as dst:
            dst.write(arr, 1)

    manifest = tmp / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["col", "row", "size", "split", "valid_frac", "label_frac"]
        )
        writer.writeheader()
        for r in range(0, IMG, CROP):
            for c in range(0, IMG, CROP):
                writer.writerow({"col": c, "row": r, "size": CROP, "split": "train",
                                 "valid_frac": 1.0, "label_frac": 1.0})

    out_dir = tmp / "ctx_cache"
    subprocess.run(
        [sys.executable, "-m", "vision_backend.prep_seg_context_cache",
         "--manifest", str(manifest), "--imagery", str(tmp / "img.tif"),
         "--context-size", "2048", "--context-output-size", "256",
         "--out-dir", str(out_dir)],
        check=True, capture_output=True,
    )
    return {"dir": tmp, "manifest": manifest, "cache": out_dir,
            "records": load_seg_records(manifest)}


def test_fixture_actually_encodes_position(scene):
    """Guards the method: if every tile looked alike, alignment would be untestable."""
    with rasterio.open(scene["dir"] / "img.tif") as src:
        a = int(src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))[0, 0])
        b = int(src.read(1, window=rasterio.windows.Window(2048, 1024, 1, 1))[0, 0])
    assert a != b
    assert a == _encode(0, 0)
    assert b == _encode(1024, 2048)


def test_every_cached_row_is_centred_on_its_own_crop(scene):
    """The core assertion: row i's context surrounds crop i, for EVERY row."""
    context = np.load(scene["cache"] / "context.npy", mmap_mode="r")
    records = scene["records"]
    assert context.shape[0] == len(records)

    mismatched = []
    for record in records:
        centre = int(context[record.index, 128, 128])
        expected = _encode(record.row + CROP // 2, record.col + CROP // 2)
        if centre != expected:
            mismatched.append((record.index, centre, expected))
    assert not mismatched, (
        f"{len(mismatched)} rows are not centred on their own crop, e.g. "
        f"{mismatched[:3]} (index, got, expected)"
    )


def test_a_reordered_cache_is_detected(scene):
    """Proves the check has teeth: permute a CORRECT cache, it must fail.

    Without this, `test_every_cached_row_is_centred_on_its_own_crop` could pass
    vacuously if the encoding or the centre-pixel lookup were wrong.
    """
    context = np.array(np.load(scene["cache"] / "context.npy", mmap_mode="r"))
    rolled = np.roll(context, shift=1, axis=0)  # every row now belongs to its neighbour
    records = scene["records"]

    mismatched = sum(
        1 for record in records
        if int(rolled[record.index, 128, 128])
        != _encode(record.row + CROP // 2, record.col + CROP // 2)
    )
    assert mismatched > 0, "a rolled cache went undetected -- the check is vacuous"


def test_dataset_serves_the_matching_context_for_each_sample(scene):
    """End-to-end through the loader, not just the raw array."""
    dataset = SegmentationCropDataset(
        scene["records"], imagery_path=scene["dir"] / "img.tif",
        label_path=scene["dir"] / "lab.tif",
        use_context=True, context_cache_dir=scene["cache"], spatial_jitter_px=0,
    )
    for i in (0, 7, 23, len(scene["records"]) - 1):
        record = scene["records"][i]
        sample = dataset[i]
        # normalized to [-1, 1] by the dataset; invert to compare with the stamp
        centre = round(float(sample["context"][0, 128, 128]) * 127.5 + 127.5)
        expected = _encode(record.row + CROP // 2, record.col + CROP // 2)
        assert abs(centre - expected) <= 1, (
            f"sample {i} (col={record.col}, row={record.row}) got context "
            f"centred on {centre}, expected {expected}"
        )


def test_live_read_matches_the_cache_exactly(scene):
    """The dev fallback must be the same window, or a smoke test proves nothing."""
    cached = SegmentationCropDataset(
        scene["records"], imagery_path=scene["dir"] / "img.tif",
        label_path=scene["dir"] / "lab.tif",
        use_context=True, context_cache_dir=scene["cache"], spatial_jitter_px=0,
    )
    live = SegmentationCropDataset(
        scene["records"], imagery_path=scene["dir"] / "img.tif",
        label_path=scene["dir"] / "lab.tif",
        use_context=True, context_size=2048, context_output_size=256,
        spatial_jitter_px=0,
    )
    for i in (0, 11, 42):
        a, b = cached[i]["context"], live[i]["context"]
        assert a.shape == b.shape
        assert float((a - b).abs().max()) == 0.0, f"sample {i}: cache != live read"


def test_full_manifest_cache_is_accepted_for_a_train_val_split(scene):
    """The regression that held the v1/v3 variant sweeps for 4 days.

    The cache is built over the WHOLE manifest and addressed by `rec.index`,
    which `partition_records` preserves. So on any manifest that has a val
    split, the cache necessarily has MORE rows than the train split has
    records -- the correct state of affairs. The old check demanded
    `count == len(records)` and rejected it: 55,702 cache rows vs a 53,971-row
    train split, every trial dead in under a second.

    The fixture's own manifest is all-train, which is exactly why the suite
    missed this; this test splits it.
    """
    records = [
        dataclasses.replace(record, split="val" if i % 8 == 0 else "train")
        for i, record in enumerate(scene["records"])
    ]
    train_records, val_records = partition_records(records)
    assert val_records, "fixture must actually produce a val split"
    assert len(train_records) < len(scene["records"])

    for split in (train_records, val_records):
        dataset = SegmentationCropDataset(
            split, imagery_path=scene["dir"] / "img.tif",
            label_path=scene["dir"] / "lab.tif",
            use_context=True, context_cache_dir=scene["cache"],
            spatial_jitter_px=0,
        )
        assert len(dataset) == len(split)
        # Accepting the cache is only right if it still pairs correctly: the
        # last val record is the one furthest from its position in the split.
        record = split[-1]
        centre = round(float(dataset[len(split) - 1]["context"][0, 128, 128]) * 127.5 + 127.5)
        expected = _encode(record.row + CROP // 2, record.col + CROP // 2)
        assert abs(centre - expected) <= 1, (
            f"split record {record.index} got context centred on {centre}, "
            f"expected {expected}"
        )


def test_cache_with_the_wrong_row_count_is_rejected(scene):
    """A cache too short to span the manifest must not load silently."""
    bad = scene["dir"] / "bad_cache"
    bad.mkdir(exist_ok=True)
    context = np.load(scene["cache"] / "context.npy", mmap_mode="r")
    np.save(bad / "context.npy", np.array(context[:-1]))  # one row short
    meta = json.loads((scene["cache"] / "context_meta.json").read_text())
    meta["count"] = context.shape[0] - 1
    (bad / "context_meta.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="context cache has"):
        SegmentationCropDataset(
            scene["records"], imagery_path=scene["dir"] / "img.tif",
            label_path=scene["dir"] / "lab.tif",
            use_context=True, context_cache_dir=bad,
        )


def test_context_window_is_centred_not_anchored(scene):
    """A window anchored at the crop's origin would be a corner view, not context."""
    record = scene["records"][10]
    window = context_window(record, 2048)
    centre_col = window.col_off + window.width / 2
    centre_row = window.row_off + window.height / 2
    assert centre_col == record.col + CROP / 2
    assert centre_row == record.row + CROP / 2


def test_builder_self_verification_passes_on_a_good_cache(scene):
    """The builder's own --verify pass must accept the cache it just wrote."""
    from vision_backend.prep_seg_context_cache import verify_cache

    failures = verify_cache(
        scene["cache"], str(scene["manifest"]), str(scene["dir"] / "img.tif"), samples=16
    )
    assert failures == 0


def test_builder_self_verification_catches_a_reordered_cache(scene, tmp_path):
    """And rejects one whose rows were permuted -- the count-check blind spot."""
    from vision_backend.prep_seg_context_cache import verify_cache

    broken = tmp_path / "rolled"
    broken.mkdir()
    context = np.array(np.load(scene["cache"] / "context.npy", mmap_mode="r"))
    np.save(broken / "context.npy", np.roll(context, shift=1, axis=0))
    (broken / "context_meta.json").write_text(
        (scene["cache"] / "context_meta.json").read_text()
    )

    failures = verify_cache(
        broken, str(scene["manifest"]), str(scene["dir"] / "img.tif"), samples=16
    )
    assert failures > 0, "a rolled cache passed verification -- the check is useless"
