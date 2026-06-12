from __future__ import annotations

import csv
from contextlib import nullcontext
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from hirise_patchloader import compute_centered_window_spec, open_image_reader, pad_patch
except ModuleNotFoundError:
    from .hirise_patchloader import (
        compute_centered_window_spec,
        open_image_reader,
        pad_patch,
    )


@dataclass(frozen=True)
class PatchRecord:
    source_image: str
    reader_backend: str
    patch_path: Optional[Path]
    context_path: Optional[Path]
    context_base_path: Optional[Path]
    context_cache_path: Optional[Path]
    local_storage: str
    context_output_size: int
    context_scale: float
    patch_name: str
    row: int
    col: int
    top: int
    left: int
    patch_size: int
    stride: int
    valid_height: int
    valid_width: int
    pad_bottom: int
    pad_right: int
    image_height: int
    image_width: int
    channels: int
    dtype: str
    pad_mode: str
    pad_value: float
    context_size: int
    context_top: int
    context_left: int
    context_valid_height: int
    context_valid_width: int
    context_pad_top: int
    context_pad_bottom: int
    context_pad_left: int
    context_pad_right: int
    invalid_fraction: float


@dataclass
class ContextPatchLoaders:
    train: DataLoader
    val: DataLoader
    test: Optional[DataLoader]
    train_dataset: "HiRISEContextPatchDataset"
    val_dataset: "HiRISEContextPatchDataset"
    test_dataset: Optional["HiRISEContextPatchDataset"]


def _use_cuda_amp(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


def _autocast_context(device: torch.device, use_amp: bool):
    if _use_cuda_amp(device, use_amp):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def _create_tqdm_progress(
    total: int,
    desc: str,
    *,
    position: int = 0,
    leave: bool = False,
):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(
        total=total,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        smoothing=0.05,
        position=position,
        leave=leave,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%"
        ),
    )


def _as_int(value: str) -> int:
    return int(value)


def _as_float(value: str) -> float:
    return float(value)


def _resolve_index_path(index_root: Path, value: str) -> Optional[Path]:
    raw = value.strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (index_root / path).resolve()
    return path


def load_patch_records(index_path: str | Path) -> list[PatchRecord]:
    index_path = Path(index_path)
    index_root = index_path.parent.resolve()
    records: list[PatchRecord] = []

    with index_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_path = _resolve_index_path(index_root, row.get("patch_path", ""))
            context_base_path = _resolve_index_path(
                index_root,
                row.get("context_base_path", ""),
            )
            context_cache_path = _resolve_index_path(
                index_root,
                row.get("context_cache_path", ""),
            )
            legacy_context_path = _resolve_index_path(
                index_root,
                row.get("context_path", ""),
            )
            if context_cache_path is None:
                context_cache_path = legacy_context_path

            source_image_path = _resolve_index_path(index_root, row["source_image"])
            context_size = _as_int(row["context_size"])
            context_output_size = _as_int(
                row.get("context_output_size") or row.get("context_size") or "0"
            )
            context_scale = _as_float(
                row.get("context_scale")
                or str(
                    max(
                        context_size / max(context_output_size, 1),
                        1.0,
                    )
                )
            )
            records.append(
                PatchRecord(
                    source_image=str(source_image_path or row["source_image"]),
                    reader_backend=row["reader_backend"],
                    patch_path=patch_path,
                    context_path=context_cache_path,
                    context_base_path=context_base_path,
                    context_cache_path=context_cache_path,
                    local_storage=row.get("local_storage", "patch_file"),
                    context_output_size=context_output_size,
                    context_scale=context_scale,
                    patch_name=row["patch_name"],
                    row=_as_int(row["row"]),
                    col=_as_int(row["col"]),
                    top=_as_int(row["top"]),
                    left=_as_int(row["left"]),
                    patch_size=_as_int(row["patch_size"]),
                    stride=_as_int(row["stride"]),
                    valid_height=_as_int(row["valid_height"]),
                    valid_width=_as_int(row["valid_width"]),
                    pad_bottom=_as_int(row["pad_bottom"]),
                    pad_right=_as_int(row["pad_right"]),
                    image_height=_as_int(row["image_height"]),
                    image_width=_as_int(row["image_width"]),
                    channels=_as_int(row["channels"]),
                    dtype=row["dtype"],
                    pad_mode=row["pad_mode"],
                    pad_value=_as_float(row.get("pad_value", "0.0")),
                    context_size=context_size,
                    context_top=_as_int(row["context_top"]),
                    context_left=_as_int(row["context_left"]),
                    context_valid_height=_as_int(row["context_valid_height"]),
                    context_valid_width=_as_int(row["context_valid_width"]),
                    context_pad_top=_as_int(row["context_pad_top"]),
                    context_pad_bottom=_as_int(row["context_pad_bottom"]),
                    context_pad_left=_as_int(row["context_pad_left"]),
                    context_pad_right=_as_int(row["context_pad_right"]),
                    invalid_fraction=_as_float(
                        row.get("invalid_fraction")
                        or row.get("black_fraction")
                        or "0.0"
                    ),
                )
            )

    return records


def split_patch_records(
    records: Sequence[PatchRecord],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[list[PatchRecord], list[PatchRecord], list[PatchRecord]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))
    n_train = max(n_total - n_val - n_test, 0)

    train_records = shuffled[:n_train]
    val_records = shuffled[n_train : n_train + n_val]
    test_records = shuffled[n_train + n_val :]

    return train_records, val_records, test_records


def group_records_by_source(
    records: Sequence[PatchRecord],
) -> dict[str, list[PatchRecord]]:
    grouped: dict[str, list[PatchRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_image, []).append(record)
    return grouped


def _normalize_unit_range(
    array: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)

    lo = float(np.percentile(finite, lower_percentile))
    hi = float(np.percentile(finite, upper_percentile))

    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)

    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim == 3:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}.")


def _resize_tensor_spatial(
    tensor: torch.Tensor,
    size: int | tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = size

    if tuple(tensor.shape[-2:]) == tuple(target_size):
        return tensor

    if mode == "nearest":
        return F.interpolate(tensor, size=target_size, mode=mode)
    return F.interpolate(
        tensor,
        size=target_size,
        mode=mode,
        align_corners=False,
    )


class HiRISEContextPatchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PatchRecord],
        *,
        dataset_backend: str = "auto",
        local_input_size: Optional[int] = None,
        context_input_size: Optional[int] = None,
        normalize: bool = True,
    ):
        self.records = list(records)
        self.dataset_backend = dataset_backend
        self.local_input_size = local_input_size
        self.context_input_size = context_input_size
        self.normalize = normalize
        self._context_base_cache: dict[str, np.ndarray] = {}
        self._reader_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _array_to_tensor(
        self,
        array: np.ndarray,
        *,
        target_size: Optional[int],
    ) -> torch.Tensor:
        if self.normalize:
            array = _normalize_unit_range(array)
        else:
            array = array.astype(np.float32, copy=False)

        tensor = torch.from_numpy(_to_chw(array)).float()
        if target_size is not None:
            tensor = _resize_tensor_spatial(
                tensor.unsqueeze(0),
                target_size,
            ).squeeze(0)
        return tensor

    def _load_tensor(
        self,
        path: Path,
        *,
        target_size: Optional[int],
    ) -> torch.Tensor:
        return self._array_to_tensor(np.load(path), target_size=target_size)

    def _get_reader(self, record: PatchRecord):
        cached = self._reader_cache.get(record.source_image)
        if cached is None:
            cached = open_image_reader(Path(record.source_image))
            self._reader_cache[record.source_image] = cached
        return cached

    def _read_local_online(self, record: PatchRecord) -> np.ndarray:
        reader = self._get_reader(record)
        patch = reader.read_region(
            record.top,
            record.left,
            record.valid_height,
            record.valid_width,
        )
        return pad_patch(
            patch,
            pad_top=0,
            pad_bottom=record.pad_bottom,
            pad_left=0,
            pad_right=record.pad_right,
            pad_mode=record.pad_mode,
            pad_value=record.pad_value,
        )

    def _read_context_online(self, record: PatchRecord) -> np.ndarray:
        reader = self._get_reader(record)
        read_top = max(record.context_top, 0)
        read_left = max(record.context_left, 0)
        patch = reader.read_region(
            read_top,
            read_left,
            record.context_valid_height,
            record.context_valid_width,
        )
        return pad_patch(
            patch,
            pad_top=record.context_pad_top,
            pad_bottom=record.context_pad_bottom,
            pad_left=record.context_pad_left,
            pad_right=record.context_pad_right,
            pad_mode=record.pad_mode,
            pad_value=record.pad_value,
        )

    def _load_context_base_array(self, record: PatchRecord) -> np.ndarray:
        if record.context_base_path is None:
            raise FileNotFoundError(
                f"No shared context base is available for {record.patch_name}."
            )

        key = str(record.context_base_path)
        cached = self._context_base_cache.get(key)
        if cached is None:
            cached = np.load(record.context_base_path)
            self._context_base_cache[key] = cached
        return cached

    def _read_context_from_shared_base(self, record: PatchRecord) -> np.ndarray:
        base = self._load_context_base_array(record)
        output_size = max(record.context_output_size, 1)

        if base.ndim == 2:
            base_height, base_width = base.shape
        else:
            base_height, base_width = base.shape[:2]

        center_y = record.top + record.patch_size // 2
        center_x = record.left + record.patch_size // 2
        center_y_base = int(round(center_y / max(record.context_scale, 1.0)))
        center_x_base = int(round(center_x / max(record.context_scale, 1.0)))
        center_y_base = min(max(center_y_base, 0), max(base_height - 1, 0))
        center_x_base = min(max(center_x_base, 0), max(base_width - 1, 0))

        (
            _requested_top,
            read_top,
            valid_height,
            pad_top,
            pad_bottom,
        ) = compute_centered_window_spec(base_height, center_y_base, output_size)
        (
            _requested_left,
            read_left,
            valid_width,
            pad_left,
            pad_right,
        ) = compute_centered_window_spec(base_width, center_x_base, output_size)

        if base.ndim == 2:
            crop = base[read_top : read_top + valid_height, read_left : read_left + valid_width]
        else:
            crop = base[
                read_top : read_top + valid_height,
                read_left : read_left + valid_width,
                :,
            ]

        return pad_patch(
            crop,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_mode=record.pad_mode,
            pad_value=record.pad_value,
        )

    def _resolve_backend(self, record: PatchRecord) -> str:
        if self.dataset_backend != "auto":
            return self.dataset_backend

        if record.context_cache_path is not None and record.context_cache_path.exists():
            return "offline_paired_context"
        if record.context_base_path is not None and record.context_base_path.exists():
            return "offline_shared_context"
        return "online_jp2"

    def _load_local_array(self, record: PatchRecord) -> np.ndarray:
        if record.patch_path is not None and record.patch_path.exists():
            return np.load(record.patch_path)
        return self._read_local_online(record)

    def _load_context_array(self, record: PatchRecord, backend: str) -> np.ndarray:
        if backend == "offline_paired_context":
            if record.context_cache_path is None or not record.context_cache_path.exists():
                raise FileNotFoundError(
                    f"Missing per-sample context cache for {record.patch_name}."
                )
            return np.load(record.context_cache_path)

        if backend == "offline_shared_context":
            return self._read_context_from_shared_base(record)

        if backend == "online_jp2":
            return self._read_context_online(record)

        raise ValueError(f"Unsupported dataset backend: {backend}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        record = self.records[index]
        backend = self._resolve_backend(record)
        local = self._array_to_tensor(
            self._load_local_array(record),
            target_size=self.local_input_size,
        )
        context = self._array_to_tensor(
            self._load_context_array(record, backend),
            target_size=self.context_input_size,
        )
        return {"local": local, "context": context, "index": index}

    def __del__(self) -> None:
        for reader in self._reader_cache.values():
            close = getattr(reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def create_context_patch_dataloaders(
    index_path: str | Path,
    *,
    batch_size: int = 8,
    dataset_backend: str = "auto",
    local_input_size: Optional[int] = None,
    context_input_size: Optional[int] = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    num_workers: int = 0,
    seed: int = 42,
) -> ContextPatchLoaders:
    records = load_patch_records(index_path)
    train_records, val_records, test_records = split_patch_records(
        records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    train_dataset = HiRISEContextPatchDataset(
        train_records,
        dataset_backend=dataset_backend,
        local_input_size=local_input_size,
        context_input_size=context_input_size,
    )
    val_dataset = HiRISEContextPatchDataset(
        val_records,
        dataset_backend=dataset_backend,
        local_input_size=local_input_size,
        context_input_size=context_input_size,
    )
    test_dataset = (
        HiRISEContextPatchDataset(
            test_records,
            dataset_backend=dataset_backend,
            local_input_size=local_input_size,
            context_input_size=context_input_size,
        )
        if test_records
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_dataset is not None
        else None
    )

    return ContextPatchLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def run_context_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = False,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    leave_progress: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)

    cuda_amp = _use_cuda_amp(device, use_amp)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if training and cuda_amp
        else None
    )
    total_loss = 0.0
    total_samples = 0
    progress = _create_tqdm_progress(
        total=len(dataloader),
        desc=progress_desc or ("train" if training else "val"),
        position=progress_position,
        leave=leave_progress,
    )

    context = torch.enable_grad if training else torch.no_grad
    try:
        with context():
            for batch in dataloader:
                local = batch["local"].to(device, non_blocking=True).float()
                context_x = batch["context"].to(device, non_blocking=True).float()
                batch_size = local.size(0)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with _autocast_context(device, use_amp):
                    outputs = model(local, context_x)
                    loss = outputs["loss"]

                if training:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if progress is not None:
                    average_loss = total_loss / max(total_samples, 1)
                    postfix = {"loss": f"{average_loss:.4f}"}
                    if training and optimizer is not None:
                        postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                    progress.set_postfix(postfix)
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return total_loss / max(total_samples, 1)


def collect_context_examples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    num_examples: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            local = batch["local"].to(device, non_blocking=True).float()
            context_x = batch["context"].to(device, non_blocking=True).float()
            outputs = model(local, context_x)
            count = min(num_examples, local.size(0))
            return {
                "original": local[:count].detach().cpu(),
                "context": context_x[:count].detach().cpu(),
                "masked_input": outputs["masked_input"][:count].detach().cpu(),
                "reconstruction": outputs["reconstruction"][:count].detach().cpu(),
                "mask": outputs["mask"][:count].detach().cpu(),
                "indices": batch["index"][:count].detach().cpu(),
            }

    raise RuntimeError("Unable to collect examples from an empty dataloader.")


def _tensor_to_hwc(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        arr = array.detach().cpu().numpy()
    else:
        arr = np.asarray(array)

    if arr.ndim == 3 and arr.shape[0] <= 4:
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    return arr.astype(np.float32, copy=False)


def reconstruct_dataset(
    model: torch.nn.Module,
    dataset: HiRISEContextPatchDataset,
    device: torch.device,
    *,
    batch_size: int = 4,
    use_amp: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    originals: list[Optional[np.ndarray]] = [None] * len(dataset)
    reconstructions: list[Optional[np.ndarray]] = [None] * len(dataset)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            local = batch["local"].to(device, non_blocking=True).float()
            context_x = batch["context"].to(device, non_blocking=True).float()
            indices = batch["index"].tolist()

            with _autocast_context(device, use_amp):
                outputs = model(local, context_x)

            for batch_idx, record_idx in enumerate(indices):
                originals[record_idx] = _tensor_to_hwc(local[batch_idx])
                reconstructions[record_idx] = _tensor_to_hwc(
                    outputs["reconstruction"][batch_idx]
                )

    return (
        [arr for arr in originals if arr is not None],
        [arr for arr in reconstructions if arr is not None],
    )


def stitch_patch_arrays(
    records: Sequence[PatchRecord],
    arrays: Sequence[np.ndarray | torch.Tensor],
) -> np.ndarray:
    if len(records) != len(arrays):
        raise ValueError("records and arrays must have the same length.")
    if not records:
        raise ValueError("No records provided for stitching.")

    first = _tensor_to_hwc(arrays[0])
    image_height = records[0].image_height
    image_width = records[0].image_width

    if first.ndim == 2:
        canvas = np.zeros((image_height, image_width), dtype=np.float32)
        weights = np.zeros((image_height, image_width), dtype=np.float32)
    else:
        channels = first.shape[2]
        canvas = np.zeros((image_height, image_width, channels), dtype=np.float32)
        weights = np.zeros((image_height, image_width, 1), dtype=np.float32)

    for record, array in zip(records, arrays):
        patch = _tensor_to_hwc(array)
        valid_patch = patch[: record.valid_height, : record.valid_width]

        y0 = record.top
        y1 = record.top + record.valid_height
        x0 = record.left
        x1 = record.left + record.valid_width

        if valid_patch.ndim == 2:
            canvas[y0:y1, x0:x1] += valid_patch
            weights[y0:y1, x0:x1] += 1.0
        else:
            canvas[y0:y1, x0:x1, :] += valid_patch
            weights[y0:y1, x0:x1, :] += 1.0

    weights = np.clip(weights, 1e-6, None)
    return canvas / weights
