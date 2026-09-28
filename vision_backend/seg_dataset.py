"""Paired (imagery, class-label) crop dataset for Stage-3 segmentation.

Reads co-registered rasters produced by the NOAH-H alignment pipeline:
  * imagery  -- HiRISE DRG warped onto the NOAH-H label grid (uint8, 0=nodata)
  * labels   -- DC/IG class-index raster on the same grid (uint8, 0=nodata)

Both share the label CRS/resolution/origin, so a window is paired by *ground
bounds*, not by assuming identical pixel indices. Imagery is dequantized to
[-1, 1] exactly like SimMIM pretraining; labels are remapped 1..C -> 0..C-1 and
nodata (0) -> ``ignore_index`` so it is excluded from the loss. Flip-only
augmentation (no rotation -- shading direction is fixed by sun azimuth).

Windows come from a manifest CSV written by ``prep_seg_crops`` (col,row,size in
imagery pixels, plus a train/val split tag).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from rasterio.windows import Window, from_bounds
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = 255


@dataclass(frozen=True)
class SegCropRecord:
    col: int
    row: int
    size: int
    split: str
    # Row position in the source manifest CSV, in `load_seg_records` order --
    # lets a cache built by `prep_seg_crop_cache.py` (written in that same
    # order) be indexed after `partition_records` has subset the list.
    index: int = -1


@dataclass
class SegLoaders:
    train: DataLoader
    val: Optional[DataLoader]
    train_dataset: "SegmentationCropDataset"
    val_dataset: Optional["SegmentationCropDataset"]
    num_classes: int


def load_seg_records(manifest_path: str | Path) -> list[SegCropRecord]:
    records: list[SegCropRecord] = []
    with Path(manifest_path).open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            records.append(
                SegCropRecord(
                    col=int(row["col"]),
                    row=int(row["row"]),
                    size=int(row["size"]),
                    split=row.get("split", "train"),
                    index=i,
                )
            )
    return records


def partition_records(records: Sequence[SegCropRecord]):
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    return train, val


def _check_cache_spans_split(
    cache_rows: int,
    records: Sequence[SegCropRecord],
    *,
    cache_kind: str,
    manifest: object = None,
) -> None:
    """Verify a `rec.index`-addressed cache covers every record in this split.

    Both caches are written one row per manifest row in `load_seg_records`
    order and read back as `cache[rec.index]` -- where `index` is the position
    in the *full* manifest, deliberately preserved through `partition_records`
    and the `train_fraction` subsample (see the note there). So the test is
    that the cache spans the highest index this split uses.

    It is emphatically NOT `cache_rows == len(records)`: a split is a subset of
    the manifest, so that only holds when the split happens to BE the whole
    manifest. Requiring it rejected a perfectly good full-manifest cache for
    every manifest carrying both a train and a val split -- which is what held
    the v1/v3 variant sweeps (cache 55,702 rows, train split 53,971 records).

    This is a bounds check, not an alignment check. A cache that is merely
    REORDERED has the right length and passes here; only a content check can
    catch that -- `prep_seg_context_cache.py --verify-only`, per
    tests/test_context_cache_alignment.py.
    """
    max_index = max((rec.index for rec in records), default=-1)
    if max_index >= cache_rows:
        raise ValueError(
            f"{cache_kind} cache has {cache_rows} rows but this split "
            f"references manifest row {max_index}; it was built from "
            f"{manifest!r}. Rows are addressed by rec.index, so a cache that "
            f"does not span the manifest would pair crops with the wrong "
            f"data. Rebuild it from the manifest this run loads."
        )


def _accumulate_class_counts(
    counts: np.ndarray,
    arr: np.ndarray,
    lab: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    target = lab.astype(np.int64) - 1
    target[lab == 0] = ignore_index
    target[arr == 0] = ignore_index
    # bincount over the known small [0, num_classes) label range is an O(n)
    # counting pass; np.unique(..., return_counts=True) sorts first, which is
    # needless overhead multiplied across tens of thousands of crops.
    valid = (target >= 0) & (target < num_classes)
    if valid.any():
        counts += np.bincount(target[valid], minlength=num_classes)[:num_classes]


def compute_class_pixel_counts(
    records: Sequence[SegCropRecord],
    *,
    imagery_path: str | Path,
    label_path: str | Path,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
    cache_dir: str | Path | None = None,
) -> dict[int, int]:
    """Ground-truth pixel count per class (0-based, post 1..C -> 0..C-1 remap)
    across `records` -- for `training.utils.compute_class_weights`.

    Mirrors SegmentationCropDataset.__getitem__'s label pairing exactly (by
    ground bounds via the imagery transform, not assumed-identical col/row) so
    the counts match what training actually sees, including its
    imagery-nodata -> ignore_index rule.

    `cache_dir` (a `prep_seg_crop_cache.py` output dir, same as
    `SegmentationCropDataset(cache_dir=...)`) reads the padded crop straight
    out of the memmapped cache arrays instead of the source GeoTIFFs -- for a
    large mosaic this avoids one windowed-read-plus-DEFLATE-decompression per
    record. Note this still has to touch on the order of one pass over the
    cache's raw (uncompressed) bytes -- on a manifest whose crops don't fit
    comfortably in RAM alongside everything else running, expect this to be
    disk-bound and take on the order of minutes; `load_or_compute_class_pixel_counts`
    memoizes the result so that cost is paid at most once per manifest.
    """
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    counts = np.zeros(num_classes, dtype=np.int64)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "meta.json").read_text())
        margin = int(meta["jitter_margin"])
        images = np.load(cache_dir / "images.npy", mmap_mode="r")
        labels = np.load(cache_dir / "labels.npy", mmap_mode="r")
        iterator = tqdm(records, desc="Class pixel counts (cached crops)", unit="crop") if tqdm else records
        for rec in iterator:
            s = rec.size
            arr = images[rec.index, margin:margin + s, margin:margin + s]
            lab = labels[rec.index, margin:margin + s, margin:margin + s]
            _accumulate_class_counts(
                counts, arr, lab, num_classes=num_classes, ignore_index=ignore_index
            )
        return {c: int(counts[c]) for c in range(num_classes)}

    with rasterio.open(str(imagery_path)) as img_ds, rasterio.open(str(label_path)) as lab_ds:
        iterator = tqdm(records, desc="Class pixel counts (raw rasters)", unit="crop") if tqdm else records
        for rec in iterator:
            img_win = Window(rec.col, rec.row, rec.size, rec.size)
            arr = img_ds.read(1, window=img_win, boundless=True, fill_value=0)

            bounds = rasterio.windows.bounds(img_win, img_ds.transform)
            lab_win = from_bounds(*bounds, transform=lab_ds.transform).round_offsets().round_lengths()
            lab = lab_ds.read(
                1, window=lab_win, out_shape=(rec.size, rec.size),
                boundless=True, fill_value=0,
            )
            _accumulate_class_counts(
                counts, arr, lab, num_classes=num_classes, ignore_index=ignore_index
            )
    return {c: int(counts[c]) for c in range(num_classes)}


def load_or_compute_class_pixel_counts(
    records: Sequence[SegCropRecord],
    *,
    imagery_path: str | Path,
    label_path: str | Path,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[int, int]:
    """`compute_class_pixel_counts`, memoized to a JSON file next to the
    manifest so repeat runs against the same manifest (different loss/
    augmentation/optimizer hyperparameters, same data) skip the per-crop pass
    entirely rather than recomputing identical counts every time.

    Keyed on the manifest path + its mtime + num_classes/ignore_index/record
    count, so an edited or regenerated manifest invalidates the cache instead
    of silently returning stale counts. `manifest_path=None` (e.g. a custom
    `--loader-factory` with no manifest file) disables caching.
    """
    if manifest_path is None:
        return compute_class_pixel_counts(
            records, imagery_path=imagery_path, label_path=label_path,
            num_classes=num_classes, ignore_index=ignore_index, cache_dir=cache_dir,
        )

    manifest_path = Path(manifest_path)

    # Per-crop histograms (scripts/resplit_seg_manifest.py) are split-independent:
    # summing the rows for whichever crops are in this split gives the counts
    # directly, so changing the train/val split costs nothing instead of forcing
    # a fresh pass over the whole crop cache.
    hist_path = manifest_path.with_suffix(".crop_hists.npy")
    if hist_path.exists():
        hists = np.load(hist_path)
        idx = [r.index for r in records]
        if hists.shape[1] >= num_classes and idx and max(idx) < hists.shape[0]:
            counts = hists[idx, :num_classes].sum(axis=0)
            print(f"[seg_dataset] Class pixel counts summed from {hist_path.name} "
                  f"({len(idx):,} crops, no cache scan).")
            return {c: int(counts[c]) for c in range(num_classes)}

    cache_path = manifest_path.with_suffix(".class_pixel_counts.json")
    manifest_mtime = manifest_path.stat().st_mtime

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("manifest_mtime") == manifest_mtime
            and cached.get("num_classes") == num_classes
            and cached.get("ignore_index") == ignore_index
            and cached.get("num_records") == len(records)
        ):
            print(f"[seg_dataset] Loaded cached class pixel counts from {cache_path}")
            return {int(k): int(v) for k, v in cached["counts"].items()}

    counts = compute_class_pixel_counts(
        records, imagery_path=imagery_path, label_path=label_path,
        num_classes=num_classes, ignore_index=ignore_index, cache_dir=cache_dir,
    )
    cache_path.write_text(json.dumps({
        "manifest_path": str(manifest_path),
        "manifest_mtime": manifest_mtime,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "num_records": len(records),
        "counts": counts,
    }, indent=2))
    print(f"[seg_dataset] Cached class pixel counts to {cache_path}")
    return counts


class SegmentationCropDataset(Dataset):
    """
    augment=True applies three independent augmentations, each safe for HiRISE
    orthoimagery specifically:

    - **Flip** (existing): horizontal/vertical only, never rotation -- shading
      direction is fixed by sun azimuth, so rotating would teach the model
      shadow/slope relationships that don't occur in real HiRISE scenes. A flip
      already isn't shading-neutral either, but it's a clean pixel remapping
      (no interpolation) and was the pipeline's existing, deliberate trade-off;
      untouched here.
    - **Spatial jitter** (new): +/-`spatial_jitter_px` random translation before
      cropping, so nearby-but-not-identical views of the same terrain enter
      training as distinct samples -- more effective diversity from the same
      manifest without needing more source data. Reads boundless (fill=0), so a
      jittered crop that spills past the image edge just becomes more nodata,
      already excluded from the loss the same way real nodata is.
    - **Photometric jitter** (new): random brightness/contrast on valid pixels
      only (invalid stays pinned at exactly -1.0, preserving the clean nodata
      signal). Directly targets illumination/dynamic-range differences between
      this training scene and whatever HiRISE observation inference later runs
      on -- a real, observed failure mode (see mars-inference's non-uint8
      quantization path), not just generic regularization.
    """

    def __init__(
        self,
        records: Sequence[SegCropRecord],
        *,
        imagery_path: str | Path,
        label_path: str | Path,
        augment: bool = False,
        ignore_index: int = IGNORE_INDEX,
        spatial_jitter_px: int = 32,
        brightness_jitter: float = 0.15,
        contrast_jitter: float = 0.15,
        cache_dir: Optional[str | Path] = None,
        use_context: bool = False,
        context_cache_dir: Optional[str | Path] = None,
        context_size: int = 2048,
        context_output_size: int = 512,
    ):
        self.records = list(records)
        self.imagery_path = str(imagery_path)
        self.label_path = str(label_path)
        self.augment = augment
        self.ignore_index = ignore_index
        self.spatial_jitter_px = spatial_jitter_px
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        # Opened lazily per worker process (rasterio datasets are not fork-safe).
        self._img = None
        self._lab = None

        # Optional padded-crop cache from prep_seg_crop_cache.py: skips the
        # rasterio window read (and, for jittered reads, the up-to-4-tile
        # DEFLATE decompression that a non-grid-aligned window forces on
        # every single access) in favor of a slice out of a memmapped array.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache_images = None
        self._cache_labels = None
        self._cache_jitter_margin = 0
        if self.cache_dir is not None:
            meta = json.loads((self.cache_dir / "meta.json").read_text())
            self._cache_jitter_margin = int(meta["jitter_margin"])
            if self.spatial_jitter_px > self._cache_jitter_margin:
                raise ValueError(
                    f"spatial_jitter_px={self.spatial_jitter_px} exceeds the "
                    f"cache's jitter_margin={self._cache_jitter_margin} "
                    f"(built from {meta.get('manifest')!r}); rebuild the cache "
                    f"with --jitter-margin >= {self.spatial_jitter_px}."
                )
            # `_read_cached` slices images[rec.index] with no bounds check of
            # its own: a cache built from a shorter manifest would either
            # IndexError deep in a worker or, worse, silently serve some other
            # crop's pixels. Same contract as the context cache below.
            _check_cache_spans_split(
                int(meta["count"]), self.records,
                cache_kind="crop", manifest=meta.get("manifest"),
            )

        # --- context branch (off by default: Phase-1 loading is untouched) ---
        # A wide window centred on the crop, downsampled, for the context-aware
        # encoder's second branch. Served from prep_seg_context_cache.py's
        # context.npy (index-aligned with this manifest), or read live -- the
        # live path is a dev convenience so a one-batch smoke runs without the
        # cache, not something to train on (it re-reads a 2048px window per item).
        self.use_context = bool(use_context)
        self.context_size = int(context_size)
        self.context_output_size = int(context_output_size)
        self.context_cache_dir = (
            Path(context_cache_dir) if context_cache_dir is not None else None
        )
        self._cache_context = None
        if self.context_cache_dir is not None:
            cmeta = json.loads(
                (self.context_cache_dir / "context_meta.json").read_text()
            )
            _check_cache_spans_split(
                int(cmeta["count"]), self.records,
                cache_kind="context", manifest=cmeta.get("manifest"),
            )
            self.context_size = int(cmeta["context_size"])
            self.context_output_size = int(cmeta["context_output_size"])

    def __len__(self) -> int:
        return len(self.records)

    def _readers(self):
        if self._img is None:
            self._img = rasterio.open(self.imagery_path)
            self._lab = rasterio.open(self.label_path)
        return self._img, self._lab

    def _cache_arrays(self):
        # mmap opened lazily per worker process, same rationale as _readers.
        if self._cache_images is None:
            self._cache_images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
            self._cache_labels = np.load(self.cache_dir / "labels.npy", mmap_mode="r")
        return self._cache_images, self._cache_labels

    def _flip_augment(self, img: np.ndarray, lab: np.ndarray):
        if torch.rand(()) < 0.5:  # horizontal flip
            img = img[:, ::-1]
            lab = lab[:, ::-1]
        if torch.rand(()) < 0.5:  # vertical flip
            img = img[::-1, :]
            lab = lab[::-1, :]
        return img, lab

    def _photometric_augment(self, x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        if self.brightness_jitter <= 0 and self.contrast_jitter <= 0:
            return x
        brightness = 1.0 + (torch.rand(()).item() * 2 - 1) * self.brightness_jitter
        contrast = 1.0 + (torch.rand(()).item() * 2 - 1) * self.contrast_jitter
        jittered = np.clip(x * contrast + (brightness - 1.0), -1.0, 1.0)
        return np.where(valid_mask, jittered, -1.0).astype(np.float32)

    def _jittered_position(self, col: int, row: int) -> tuple[int, int]:
        if not self.augment or self.spatial_jitter_px <= 0:
            return col, row
        j = self.spatial_jitter_px
        dc = int(torch.randint(-j, j + 1, (1,)).item())
        dr = int(torch.randint(-j, j + 1, (1,)).item())
        return col + dc, row + dr

    def _read_live(self, rec: SegCropRecord, S: int) -> tuple[np.ndarray, np.ndarray]:
        img_ds, lab_ds = self._readers()
        col, row = self._jittered_position(rec.col, rec.row)

        img_win = Window(col, row, S, S)
        # boundless: a jittered window may spill past the image edge; those
        # pixels come back 0 (nodata), same as real off-swath imagery.
        arr = img_ds.read(1, window=img_win, boundless=True, fill_value=0)  # uint8 (S, S)

        # Pair the label window by ground bounds so any grid offset is handled.
        bounds = rasterio.windows.bounds(img_win, img_ds.transform)
        lab_win = from_bounds(*bounds, transform=lab_ds.transform).round_offsets().round_lengths()
        lab = lab_ds.read(
            1, window=lab_win, out_shape=(S, S),
            boundless=True, fill_value=0,
        )
        return arr, lab

    def _read_cached(self, rec: SegCropRecord, S: int) -> tuple[np.ndarray, np.ndarray]:
        images, labels = self._cache_arrays()
        j = self._cache_jitter_margin
        dc, dr = 0, 0
        if self.augment and self.spatial_jitter_px > 0:
            jp = self.spatial_jitter_px
            dc = int(torch.randint(-jp, jp + 1, (1,)).item())
            dr = int(torch.randint(-jp, jp + 1, (1,)).item())
        r0, c0 = j + dr, j + dc
        arr = images[rec.index, r0:r0 + S, c0:c0 + S]
        lab = labels[rec.index, r0:r0 + S, c0:c0 + S]
        return arr, lab

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        S = rec.size

        if self.cache_dir is not None:
            arr, lab = self._read_cached(rec, S)
        else:
            arr, lab = self._read_live(rec, S)
        lab = lab.astype(np.int64)

        x = arr.astype(np.float32) / 127.5 - 1.0  # -> [-1, 1]
        # remap class ids 1..C -> 0..C-1; nodata (0) and imagery-nodata -> ignore
        target = lab - 1
        target[lab == 0] = self.ignore_index
        target[arr == 0] = self.ignore_index

        if self.augment:
            x, target = self._flip_augment(x, target)
            x = self._photometric_augment(x, valid_mask=(x != -1.0))

        image = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)  # (1, S, S)
        label = torch.from_numpy(np.ascontiguousarray(target)).long()   # (S, S)
        sample = {"image": image, "label": label, "index": index}

        if self.use_context:
            ctx = self._read_context(rec)
            # Normalized exactly like the local crop, so both branches see the
            # same input distribution.
            ctx = ctx.astype(np.float32) / 127.5 - 1.0
            sample["context"] = torch.from_numpy(
                np.ascontiguousarray(ctx)
            ).unsqueeze(0)  # (1, O, O)
        return sample

    def _read_context(self, rec: SegCropRecord) -> np.ndarray:
        """The downsampled context window for this record, cache or live."""
        if self.context_cache_dir is not None:
            if self._cache_context is None:
                self._cache_context = np.load(
                    self.context_cache_dir / "context.npy", mmap_mode="r"
                )
            # Addressed by rec.index, which is why the cache must be built from
            # this same manifest -- see prep_seg_context_cache.py.
            return np.asarray(self._cache_context[rec.index])
        return self._read_context_live(rec)

    def _read_context_live(self, rec: SegCropRecord) -> np.ndarray:
        """Dev fallback: read the context window straight from the raster.

        Lets a one-batch smoke run before the cache exists. Far too slow to
        train on -- it re-reads and downsamples a `context_size` window per item.
        """
        try:
            from vision_backend.prep_seg_context_cache import context_window
        except ModuleNotFoundError:
            from prep_seg_context_cache import context_window

        img_ds, _ = self._readers()
        win = context_window(rec, self.context_size)
        return img_ds.read(
            1, window=win,
            out_shape=(self.context_output_size, self.context_output_size),
            boundless=True, fill_value=0,
            resampling=Resampling.average,
        )

    def __del__(self):
        for ds in (self._img, self._lab):
            try:
                if ds is not None:
                    ds.close()
            except Exception:
                pass


def _make_loader(dataset, *, batch_size, shuffle, num_workers, persistent_workers,
                 prefetch_factor, pin_memory, generator):
    kwargs: dict = dict(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=shuffle,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    if shuffle and generator is not None:
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def create_segmentation_dataloaders(
    manifest_path: str | Path,
    *,
    imagery_path: str | Path,
    label_path: str | Path,
    num_classes: int,
    batch_size: int = 8,
    num_workers: int = 0,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    pin_memory: bool = True,
    augment: bool = True,
    spatial_jitter_px: int = 32,
    brightness_jitter: float = 0.15,
    contrast_jitter: float = 0.15,
    seed: int = 42,
    cache_dir: Optional[str | Path] = None,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    use_context: bool = False,
    context_cache_dir: Optional[str | Path] = None,
    context_size: int = 2048,
    context_output_size: int = 512,
) -> SegLoaders:
    records = load_seg_records(manifest_path)
    train_records, val_records = partition_records(records)

    # Subsample *in memory*, never by rewriting the manifest: SegCropRecord.index
    # is the row into the padded crop cache (images.npy/labels.npy), so dropping
    # rows from the CSV would renumber every record and silently pair crops with
    # the wrong labels. Selecting a subset of the loaded records keeps each
    # record's original .index, so the cache stays correct.
    def _subsample(recs, frac, tag):
        if frac >= 1.0:
            return recs
        if not 0.0 < frac < 1.0:
            raise ValueError(f"{tag}_fraction must be in (0, 1]; got {frac}")
        rng = np.random.default_rng(seed)
        keep = rng.permutation(len(recs))[: max(1, int(round(len(recs) * frac)))]
        subset = [recs[i] for i in sorted(keep.tolist())]
        print(f"[seg_dataset] {tag}: {len(recs):,} -> {len(subset):,} crops "
              f"(fraction={frac}, seed={seed}); crop-cache indices preserved.")
        return subset

    train_records = _subsample(train_records, float(train_fraction), "train")
    val_records = _subsample(val_records, float(val_fraction), "val")

    context_kwargs = dict(
        use_context=use_context, context_cache_dir=context_cache_dir,
        context_size=context_size, context_output_size=context_output_size,
    )
    train_dataset = SegmentationCropDataset(
        train_records, imagery_path=imagery_path, label_path=label_path, augment=augment,
        spatial_jitter_px=spatial_jitter_px, brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter, cache_dir=cache_dir, **context_kwargs,
    )
    val_dataset = (
        SegmentationCropDataset(
            val_records, imagery_path=imagery_path, label_path=label_path, augment=False,
            cache_dir=cache_dir, **context_kwargs,
        )
        if val_records
        else None
    )

    generator = torch.Generator().manual_seed(seed)
    pin = pin_memory and torch.cuda.is_available()

    train_loader = _make_loader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        pin_memory=pin, generator=generator,
    )
    val_loader = (
        _make_loader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
            pin_memory=pin, generator=None,
        )
        if val_dataset is not None
        else None
    )

    return SegLoaders(
        train=train_loader, val=val_loader,
        train_dataset=train_dataset, val_dataset=val_dataset,
        num_classes=num_classes,
    )
