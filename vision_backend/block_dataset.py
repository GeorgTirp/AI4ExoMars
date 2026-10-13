"""Block-level labels: one class per 512x512 block (planetary-scientist labels).

Manifest CSV, one row per block:

    label   class name (a NOAH-H DC name, matched case/spacing-insensitively and
            tolerant of the legend's "Continous" spelling) or a DC id 1..14;
            anything else is a NEW class
    split   train / val / test -- assign it spatially (whole regions or HiRISE
            images per split): neighbouring blocks share terrain, so a random
            block-level split leaks and overstates accuracy
    either  col,row   top-left pixel of the block in --imagery-path (the DRG on
                      the NOAH-H label grid, same as the segmentation crops)
    or      image     path to a single-band square uint8 tile, side a multiple
                      of 512 (a 1024 block is classified whole: 4 backbone
                      tiles, one pooled prediction), quantised like the
                      mosaic (0 = nodata) -- scripts/prep_expert_block_manifest.py
    optional block_id

Class order for a run: all 14 NOAH-H classes in DC-id order (present in the
labels or not -- the block head's prior uses the segmenter's full class set),
then the new classes sorted by name. Imagery is dequantised to [-1, 1] exactly
as in the segmentation dataset.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_DC_CLASSES_JSON = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/classes_DC.json"


@dataclass(frozen=True)
class BlockRecord:
    label: int
    split: str
    col: Optional[int] = None
    row: Optional[int] = None
    image: Optional[str] = None
    block_id: str = ""
    # spatial cross-validation (prep_expert_block_manifest.py --cv-folds): this
    # block's fold, and the folds whose held-out blocks it borders
    fold: Optional[int] = None
    cv_buffer: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlockClasses:
    names: list[str]
    # segmenter DC index (0-based) per block class, None for new classes
    seg_index: list[Optional[int]]

    @property
    def num_new(self) -> int:
        return sum(s is None for s in self.seg_index)


def normalize_class_name(name: str) -> str:
    s = re.sub(r"\s+", " ", name.strip().lower())
    return s.replace("continous", "continuous")


def resolve_block_classes(raw_labels: Sequence[str], dc_classes_json: str | Path) -> BlockClasses:
    dc = json.loads(Path(dc_classes_json).read_text())["classes"]
    dc = sorted(dc, key=lambda c: int(c["id"]))
    names = [c["name"] for c in dc]
    seg_index: list[Optional[int]] = list(range(len(dc)))
    by_name = {normalize_class_name(c["name"]): i for i, c in enumerate(dc)}
    by_id = {str(int(c["id"])): i for i, c in enumerate(dc)}
    new = sorted({lab.strip() for lab in raw_labels
                  if lab.strip() not in by_id and normalize_class_name(lab) not in by_name})
    return BlockClasses(names=names + new, seg_index=seg_index + [None] * len(new))


def label_to_index(label: str, classes: BlockClasses) -> int:
    lab = label.strip()
    num_dc = len(classes.names) - classes.num_new
    if lab.isdigit() and 1 <= int(lab) <= num_dc:
        return int(lab) - 1  # DC id 1..14 -> 0..13 (the NOAH-H classes come first)
    norm = normalize_class_name(lab)
    for i, name in enumerate(classes.names):
        if (i < num_dc and normalize_class_name(name) == norm) or name == lab:
            return i
    raise KeyError(f"label {label!r} is not in the class list {classes.names}")


def load_block_classes(path: str | Path) -> BlockClasses:
    """A fixed class list ({"classes": [{"name", "seg_index"}]}, written by
    scripts/prep_expert_block_manifest.py) -- keeps the head's classes identical
    across labelling campaigns, whichever classes a campaign happens to contain."""
    entries = json.loads(Path(path).read_text())["classes"]
    seg_index = [None if e["seg_index"] is None else int(e["seg_index"]) for e in entries]
    num_dc = sum(s is not None for s in seg_index)
    if seg_index != list(range(num_dc)) + [None] * (len(seg_index) - num_dc):
        raise ValueError(f"{path}: NOAH-H classes must come first, in segmenter order, then the new ones")
    return BlockClasses(names=[e["name"] for e in entries], seg_index=seg_index)


def load_block_manifest(
    manifest_path: str | Path,
    dc_classes_json: str | Path = DEFAULT_DC_CLASSES_JSON,
    block_classes: Optional[str | Path] = None,
) -> tuple[list[BlockRecord], BlockClasses]:
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{manifest_path}: empty manifest")
    cols = set(rows[0])
    if not {"label", "split"} <= cols or not ({"col", "row"} <= cols or "image" in cols):
        raise ValueError(f"{manifest_path}: need columns label, split and col,row or image; got {sorted(cols)}")

    if block_classes is not None:
        classes = load_block_classes(block_classes)
    else:
        classes = resolve_block_classes([r["label"] for r in rows], dc_classes_json)
    records = []
    for i, r in enumerate(rows):
        image = (r.get("image") or "").strip() or None
        if image is not None and not Path(image).is_absolute():
            image = str(manifest_path.parent / image)
        records.append(BlockRecord(
            label=label_to_index(r["label"], classes),
            split=r["split"].strip().lower(),
            col=int(r["col"]) if image is None else None,
            row=int(r["row"]) if image is None else None,
            image=image,
            block_id=(r.get("block_id") or str(i)).strip(),
            fold=int(r["fold"]) if (r.get("fold") or "").strip() else None,
            cv_buffer=tuple(int(f) for f in (r.get("cv_buffer") or "").split(";") if f.strip()),
        ))
    return records, classes


class BlockDataset(Dataset):
    """Returns {"image": [1,S,S] in [-1,1], "valid": [1,S,S] bool, "label": int}.

    Blocks are read once into memory (a few thousand 576^2 uint8 windows is a
    few GB at most). Mosaic blocks are read with a `jitter` margin so training
    can shift the window by up to +/-jitter px -- the block label still holds
    for a 6 % shift -- plus the segmentation recipe's flips and brightness /
    contrast jitter. Tile files have no surrounding imagery, so no shift.
    """

    def __init__(
        self,
        records: Sequence[BlockRecord],
        *,
        imagery_path: Optional[str | Path] = None,
        block_size: int = 512,
        augment: bool = False,
        jitter: int = 32,
        brightness_jitter: float = 0.15,
        contrast_jitter: float = 0.15,
    ):
        import rasterio
        from rasterio.windows import Window

        self.records = list(records)
        self.size = int(block_size)
        self.augment = augment
        self.jitter = int(jitter)
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        S, j = self.size, self.jitter
        self.arrays: list[np.ndarray] = []
        self.margins: list[int] = []  # stored jitter margin per block (0 for tile files)
        mosaic = None
        try:
            for rec in self.records:
                if rec.image is not None:
                    with rasterio.open(rec.image) as ds:
                        arr = ds.read(1)
                    if arr.dtype != np.uint8 or arr.shape[0] != arr.shape[1] or arr.shape[0] % S:
                        raise ValueError(f"{rec.image}: expected a square uint8 tile with side a multiple "
                                         f"of {S}, got {arr.dtype} {arr.shape}")
                    self.margins.append(0)
                else:
                    self.margins.append(j)
                    if mosaic is None:
                        if imagery_path is None:
                            raise ValueError("col,row blocks need imagery_path")
                        mosaic = rasterio.open(imagery_path)
                    arr = mosaic.read(1, window=Window(rec.col - j, rec.row - j, S + 2 * j, S + 2 * j),
                                      boundless=True, fill_value=0)
                self.arrays.append(np.ascontiguousarray(arr, dtype=np.uint8))
        finally:
            if mosaic is not None:
                mosaic.close()

    def __len__(self) -> int:
        return len(self.records)

    def block_sides(self) -> list[int]:
        return [a.shape[0] - 2 * m for a, m in zip(self.arrays, self.margins)]

    def __getitem__(self, index: int) -> dict:
        arr = self.arrays[index]
        m = self.margins[index]
        S = arr.shape[0] - 2 * m
        dr = dc = m
        if self.augment and m > 0:
            dr, dc = (int(v) for v in torch.randint(0, 2 * m + 1, (2,)))
        arr = arr[dr:dr + S, dc:dc + S]
        valid = arr != 0
        x = arr.astype(np.float32) / 127.5 - 1.0
        if self.augment:
            if torch.rand(()) < 0.5:
                x, valid = x[:, ::-1], valid[:, ::-1]
            if torch.rand(()) < 0.5:
                x, valid = x[::-1, :], valid[::-1, :]
            brightness = 1.0 + (torch.rand(()).item() * 2 - 1) * self.brightness_jitter
            contrast = 1.0 + (torch.rand(()).item() * 2 - 1) * self.contrast_jitter
            x = np.where(valid, np.clip(x * contrast + (brightness - 1.0), -1.0, 1.0), -1.0)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0),
            "label": int(self.records[index].label),
            "index": index,
        }


class SizeGroupedBatchSampler(torch.utils.data.Sampler):
    """Batches of equal-sized blocks (they cannot be stacked otherwise). A block
    of side s counts (s / base)^2 toward `batch_size`, so a batch costs about
    the same GPU memory whatever the block size: 8 blocks of 512 or 2 of 1024.
    Shuffles within each size and the batch order (global torch RNG)."""

    def __init__(self, sides: Sequence[int], batch_size: int, *, base: int = 512, shuffle: bool = False):
        self.shuffle = shuffle
        self.groups: dict[int, list[int]] = {}
        for i, s in enumerate(sides):
            self.groups.setdefault(int(s), []).append(i)
        self.per_batch = {s: max(1, batch_size // max(1, (s // base) ** 2)) for s in self.groups}

    def __len__(self) -> int:
        return sum(-(-len(idx) // self.per_batch[s]) for s, idx in self.groups.items())

    def __iter__(self):
        batches = []
        for s, idx in sorted(self.groups.items()):
            order = [idx[i] for i in torch.randperm(len(idx))] if self.shuffle else idx
            per = self.per_batch[s]
            batches += [order[i:i + per] for i in range(0, len(order), per)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches))]
        return iter(batches)
