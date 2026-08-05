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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
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
        for row in csv.DictReader(f):
            records.append(
                SegCropRecord(
                    col=int(row["col"]),
                    row=int(row["row"]),
                    size=int(row["size"]),
                    split=row.get("split", "train"),
                )
            )
    return records


def partition_records(records: Sequence[SegCropRecord]):
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    return train, val


class SegmentationCropDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SegCropRecord],
        *,
        imagery_path: str | Path,
        label_path: str | Path,
        augment: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.records = list(records)
        self.imagery_path = str(imagery_path)
        self.label_path = str(label_path)
        self.augment = augment
        self.ignore_index = ignore_index
        # Opened lazily per worker process (rasterio datasets are not fork-safe).
        self._img = None
        self._lab = None

    def __len__(self) -> int:
        return len(self.records)

    def _readers(self):
        if self._img is None:
            self._img = rasterio.open(self.imagery_path)
            self._lab = rasterio.open(self.label_path)
        return self._img, self._lab

    def _augment(self, img: np.ndarray, lab: np.ndarray):
        if torch.rand(()) < 0.5:  # horizontal flip
            img = img[:, ::-1]
            lab = lab[:, ::-1]
        if torch.rand(()) < 0.5:  # vertical flip
            img = img[::-1, :]
            lab = lab[::-1, :]
        return img, lab

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        img_ds, lab_ds = self._readers()
        S = rec.size

        img_win = Window(rec.col, rec.row, S, S)
        arr = img_ds.read(1, window=img_win)  # uint8 (S, S)

        # Pair the label window by ground bounds so any grid offset is handled.
        bounds = rasterio.windows.bounds(img_win, img_ds.transform)
        lab_win = from_bounds(*bounds, transform=lab_ds.transform).round_offsets().round_lengths()
        lab = lab_ds.read(
            1, window=lab_win, out_shape=(S, S),
            boundless=True, fill_value=0,
        ).astype(np.int64)

        x = arr.astype(np.float32) / 127.5 - 1.0  # -> [-1, 1]
        # remap class ids 1..C -> 0..C-1; nodata (0) and imagery-nodata -> ignore
        target = lab - 1
        target[lab == 0] = self.ignore_index
        target[arr == 0] = self.ignore_index

        if self.augment:
            x, target = self._augment(x, target)

        image = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)  # (1, S, S)
        label = torch.from_numpy(np.ascontiguousarray(target)).long()   # (S, S)
        return {"image": image, "label": label, "index": index}

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
    seed: int = 42,
) -> SegLoaders:
    records = load_seg_records(manifest_path)
    train_records, val_records = partition_records(records)

    train_dataset = SegmentationCropDataset(
        train_records, imagery_path=imagery_path, label_path=label_path, augment=augment
    )
    val_dataset = (
        SegmentationCropDataset(
            val_records, imagery_path=imagery_path, label_path=label_path, augment=False
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
