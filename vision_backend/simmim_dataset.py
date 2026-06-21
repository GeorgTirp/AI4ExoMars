"""Single-branch crop dataset + dataloaders for Stage-1 SimMIM.

Reads the ``crops_index.csv`` manifest and ``splits.json`` written by
``prep_crops``, loads uint8 crops, dequantizes to [-1, 1], applies flip-only
augmentation (NO rotation -- shading direction is fixed by sun azimuth, §4.6),
and returns ``(1, H, W)`` float tensors.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS_FILENAME = "splits.json"


@dataclass(frozen=True)
class CropRecord:
    crop_path: Path
    observation_id: str
    index: int


@dataclass
class SimMIMLoaders:
    train: DataLoader
    val: Optional[DataLoader]
    train_dataset: "SimMIMCropDataset"
    val_dataset: Optional["SimMIMCropDataset"]


def load_crop_records(index_path: str | Path) -> list[CropRecord]:
    index_path = Path(index_path)
    root = index_path.parent
    records: list[CropRecord] = []
    with index_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            crop_path = (root / row["crop_path"]).resolve()
            records.append(
                CropRecord(
                    crop_path=crop_path,
                    observation_id=row["observation_id"],
                    index=i,
                )
            )
    return records


def load_splits(splits_path: str | Path) -> dict:
    with Path(splits_path).open() as f:
        return json.load(f)


def partition_records(
    records: Sequence[CropRecord],
    splits: dict,
) -> tuple[list[CropRecord], list[CropRecord]]:
    """Split crops by observation membership (never by crop)."""
    train_obs = set(splits.get("train", []))
    val_obs = set(splits.get("val", []))
    train = [r for r in records if r.observation_id in train_obs]
    val = [r for r in records if r.observation_id in val_obs]
    return train, val


class SimMIMCropDataset(Dataset):
    def __init__(self, records: Sequence[CropRecord], *, augment: bool = False):
        self.records = list(records)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # flips only -- no rotation (guardrail §4.6)
        if torch.rand(()) < 0.5:
            x = x[:, ::-1]
        if torch.rand(()) < 0.5:
            x = x[::-1, :]
        return x

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        arr = np.load(record.crop_path)  # uint8 (H, W)
        x = arr.astype(np.float32) / 127.5 - 1.0  # -> [-1, 1]
        if self.augment:
            x = self._augment(x)
        tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)  # (1, H, W)
        return {"image": tensor, "index": record.index}


class ResumableLoader:
    """Infinite, exactly-resumable batch iterator over a crop dataset.

    Determinism comes from a per-epoch ``randperm`` seeded by ``seed + epoch``.
    Resume restores ``(epoch, consumed_batches)`` and the global RNG, so the
    loss trajectory is bitwise-reproducible (test §8.6). Workers/prefetch are
    supported; the per-epoch DataLoader rebuild cost is negligible at scale.
    """

    def __init__(
        self,
        dataset: SimMIMCropDataset,
        *,
        batch_size: int,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = 2,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.epoch = 0
        self.consumed_batches = 0

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "consumed_batches": self.consumed_batches}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self.consumed_batches = int(state["consumed_batches"])

    def _epoch_batches(self) -> list[list[int]]:
        n = len(self.dataset)
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(n, generator=g).tolist()
        batches = [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self):
        while True:
            batches = self._epoch_batches()[self.consumed_batches :]
            # Give the DataLoader its own generator: its iterator constructor
            # draws one int64 for worker seeding, and we must not let that draw
            # come from the global RNG (which the masker uses) -- otherwise
            # rebuilding the loader on resume desyncs the mask RNG by one draw
            # and breaks bitwise-equal resume.
            loader_generator = torch.Generator().manual_seed(self.seed + self.epoch + 1)
            loader_kwargs: dict = dict(
                batch_sampler=batches,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                generator=loader_generator,
            )
            if self.num_workers > 0 and self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader = DataLoader(self.dataset, **loader_kwargs)
            for batch in loader:
                self.consumed_batches += 1
                yield batch
            self.epoch += 1
            self.consumed_batches = 0


def _make_loader(
    dataset: SimMIMCropDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    pin_memory: bool,
    generator: Optional[torch.Generator],
) -> DataLoader:
    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    if shuffle and generator is not None:
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def create_simmim_dataloaders(
    index_path: str | Path,
    *,
    splits_path: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    pin_memory: bool = True,
    augment: bool = True,
    seed: int = 42,
) -> SimMIMLoaders:
    index_path = Path(index_path)
    if splits_path is None:
        splits_path = index_path.parent / SPLITS_FILENAME

    records = load_crop_records(index_path)
    splits = load_splits(splits_path)
    train_records, val_records = partition_records(records, splits)

    train_dataset = SimMIMCropDataset(train_records, augment=augment)
    val_dataset = SimMIMCropDataset(val_records, augment=False) if val_records else None

    generator = torch.Generator().manual_seed(seed)
    pin = pin_memory and torch.cuda.is_available()

    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin,
        generator=generator,
    )
    val_loader = (
        _make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin,
            generator=None,
        )
        if val_dataset is not None
        else None
    )

    return SimMIMLoaders(
        train=train_loader,
        val=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
