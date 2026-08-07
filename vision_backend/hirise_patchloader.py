#!/usr/bin/env python3
"""
Extract fixed-size patches from HiRISE JPEG2000 images.

The script scans a directory for ``.jp2`` / ``.JP2`` images, creates one
output folder per source image, and writes a full patch grid with border
padding so every saved patch has the same size. A CSV manifest is written for
each image plus a global index for downstream sliding-window training or
reassembly. It also saves a small preview set per image so the tiling can be
inspected quickly.

Reader priority:
1. ``rasterio`` for efficient windowed reads on large images
2. ``glymur``
3. ``Pillow``
4. ``imageio.v3``

By default patches are saved as ``.npy`` arrays to preserve source dtype.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import fnmatch
from multiprocessing import Manager
import os
from queue import Empty
import sys
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "hirise_context_pairs"
INDEX_FILENAME = "patch_index.csv"
MANIFEST_FILENAME = "manifest.csv"
EXAMPLES_DIRNAME = "examples"
EXAMPLES_MANIFEST_FILENAME = "examples_manifest.csv"
CONTEXTS_DIRNAME = "contexts"
CONTEXT_BASES_DIRNAME = "context_bases"
CONTEXT_CACHE_DIRNAME = "context_cache"
DEFAULT_MASK_STRIPE_HEIGHT = 2048
MAX_INTEGRAL_IMAGE_PIXELS = 200_000_000
DEFAULT_PARALLEL_PROGRESS_UPDATE_INTERVAL = 64
DEFAULT_PROGRESS_FLUSH_SECONDS = 0.35


@dataclass(frozen=True)
class PatchSpec:
    row: int
    col: int
    top: int
    left: int
    valid_height: int
    valid_width: int
    pad_bottom: int
    pad_right: int


@dataclass(frozen=True)
class WindowSpec:
    requested_top: int
    requested_left: int
    read_top: int
    read_left: int
    valid_height: int
    valid_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


@dataclass(frozen=True)
class ImageProcessingResult:
    image_path: str
    manifest_path: str
    kept: int
    skipped_invalid: int


@dataclass(frozen=True)
class ImageInventory:
    image_path: Path
    reader_backend: str
    height: int
    width: int
    channels: int
    dtype: str
    total_candidates: int
    analyzed_candidates: int


class LocalProgressReporter:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def put(self, count: int) -> None:
        if self.progress_bar is not None and count > 0:
            self.progress_bar.update(int(count))


class BaseImageReader:
    backend = "unknown"

    def __init__(self, path: Path):
        self.path = path
        self.height = 0
        self.width = 0
        self.channels = 0
        self.dtype = "unknown"

    def read_region(self, top: int, left: int, height: int, width: int):
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RasterioImageReader(BaseImageReader):
    backend = "rasterio"

    def __init__(self, path: Path):
        super().__init__(path)
        import rasterio
        from rasterio.windows import Window

        self._window_type = Window
        self._dataset = rasterio.open(path)
        self.height = int(self._dataset.height)
        self.width = int(self._dataset.width)
        self.channels = int(self._dataset.count)
        self.dtype = str(self._dataset.dtypes[0])

    def read_region(self, top: int, left: int, height: int, width: int):
        data = self._dataset.read(
            window=self._window_type(left, top, width, height)
        )
        return _normalize_image_array(data)

    def close(self) -> None:
        self._dataset.close()


class ArrayImageReader(BaseImageReader):
    def __init__(self, path: Path, array, backend: str):
        super().__init__(path)
        self.backend = backend
        self._array = _normalize_image_array(array)

        if self._array.ndim == 2:
            self.height, self.width = self._array.shape
            self.channels = 1
        elif self._array.ndim == 3:
            self.height, self.width, self.channels = self._array.shape
        else:
            raise ValueError(
                f"Expected a 2D or 3D image array, got shape {self._array.shape}."
            )

        self.dtype = str(self._array.dtype)

    def read_region(self, top: int, left: int, height: int, width: int):
        if self._array.ndim == 2:
            return self._array[top : top + height, left : left + width]
        return self._array[top : top + height, left : left + width, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create padded fixed-size patches from HiRISE JP2 images."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory that contains the source .jp2/.JP2 files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where patch folders and manifests will be written.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2024,
        help="Square local patch size in pixels.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=2048,
        help="Square context window size in source-image pixels.",
    )
    parser.add_argument(
        "--context-output-size",
        type=int,
        default=None,
        help=(
            "Spatial size for precomputed context tensors. Defaults to the "
            "local patch size when omitted."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Patch stride. Defaults to the patch size for non-overlapping tiles.",
    )
    parser.add_argument(
        "--pad-mode",
        choices=("edge", "constant", "symmetric"),
        default="edge",
        help="Padding mode for incomplete border patches.",
    )
    parser.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
        help="Constant pad value, used only when --pad-mode=constant.",
    )
    parser.add_argument(
        "--glob",
        default="*.jp2",
        help="Filename glob used to discover images recursively.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search the input directory recursively.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help=(
            "How many source images to preprocess in parallel. "
            "Use 1 to keep processing serial."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing patch files instead of skipping them.",
    )
    parser.add_argument(
        "--disable-shared-context-base",
        action="store_true",
        help=(
            "Skip writing the shared downsampled context base per source image. "
            "This keeps only local patches plus manifest metadata."
        ),
    )
    parser.add_argument(
        "--write-context-cache",
        action="store_true",
        help=(
            "Also write per-sample precomputed context tensors at "
            "--context-output-size for maximum training throughput."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned patch counts without writing any files.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="How many example preview patches to save per source image.",
    )
    parser.add_argument(
        "--max-patches-per-image",
        type=int,
        default=25000,
        help=(
            "Maximum number of candidate patches to analyze per image. "
            "When an image exceeds this, the candidates are subsampled evenly "
            "across the scan order. Set to a negative value to disable the cap."
        ),
    )
    parser.add_argument(
        "--max-invalid-fraction",
        type=float,
        default=0.01,
        help=(
            "Drop patches whose valid region contains more than this fraction "
            "of invalid pixels. Invalid pixels are the saturation / NULL "
            "markers that map to 0 (NULL / LRS / LIS) or 255 (HIS / HRS) in "
            "the 8-bit encoding. Set to a negative value to disable filtering."
        ),
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires numpy. Install numpy plus one JP2-capable "
            "reader such as rasterio, glymur, Pillow, or imageio."
        ) from exc
    return np


def create_progress_bar(
    total: int,
    desc: str,
    *,
    unit: str = "patch",
    position: int = 0,
    leave: bool = True,
):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        position=position,
        leave=leave,
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%"
        ),
    )


def select_example_indices(num_items: int, count: int) -> list[int]:
    if count <= 0 or num_items <= 0:
        return []

    if count >= num_items:
        return list(range(num_items))

    if count == 1:
        return [num_items // 2]

    selected: list[int] = []
    seen: set[int] = set()

    for i in range(count):
        index = round(i * (num_items - 1) / (count - 1))
        if index not in seen:
            selected.append(index)
            seen.add(index)

    if len(selected) < count:
        for index in range(num_items):
            if index not in seen:
                selected.append(index)
                seen.add(index)
            if len(selected) == count:
                break

    return sorted(selected)


def preview_filename(
    image_stem: str,
    example_index: int,
    row: int,
    col: int,
    top: int,
    left: int,
) -> str:
    return (
        f"example_{example_index:02d}_"
        f"{image_stem}"
        f"_r{row:04d}"
        f"_c{col:04d}"
        f"_y{top:07d}"
        f"_x{left:07d}.png"
    )


def preview_manifest_fieldnames() -> list[str]:
    return [
        "source_image",
        "reader_backend",
        "preview_path",
        "patch_path",
        "context_path",
        "context_base_path",
        "context_cache_path",
        "example_index",
        "row",
        "col",
        "top",
        "left",
        "valid_height",
        "valid_width",
        "pad_bottom",
        "pad_right",
        "image_height",
        "image_width",
        "channels",
        "dtype",
        "pad_mode",
        "invalid_fraction",
    ]


def prepare_preview_array(array):
    np = require_numpy()

    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    min_value = float(np.percentile(finite, 1.0))
    max_value = float(np.percentile(finite, 99.0))
    if max_value <= min_value:
        min_value = float(finite.min())
        max_value = float(finite.max())

    if max_value <= min_value:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - min_value) / (max_value - min_value)
    arr = (255.0 * arr).clip(0.0, 255.0).astype(np.uint8)
    return arr


def save_preview_image(output_path: Path, patch) -> None:
    preview = prepare_preview_array(patch)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image

        Image.fromarray(preview).save(output_path)
        return
    except ModuleNotFoundError:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, preview)
        return
    except ModuleNotFoundError:
        pass

    raise RuntimeError(
        "Saving preview images requires Pillow or imageio in addition to numpy."
    )


def compute_invalid_fraction(array) -> float:
    invalid_mask = compute_invalid_mask(array)
    return float(invalid_mask.mean())


def _pixel_invalid(arr):
    """Per-pixel (per-channel) invalid mask, keyed on the array's dtype.

    HiRISE encodes NULL / saturation as special DN values that differ by bit
    depth (see the PDS product summary):

      * 8-bit  : 0 = NULL / LRS / LIS (black), 255 = HIS / HRS (white).
      * 16-bit : NULL = 0 in the products we have. The signed 16-bit markers
                 (-32768 NULL .. -32764 HRS) are NOT used by the current
                 archive, so only 0 is treated as invalid here; add them to
                 ``markers`` if a product using that convention appears.
                 (Do NOT treat 255 as invalid -- it is a normal interior DN.)
      * float  : NULL / saturation are -FLT_MAX (-3.40282e+38); also treat any
                 non-finite value as invalid.
    """
    np = require_numpy()

    if np.issubdtype(arr.dtype, np.floating):
        return (~np.isfinite(arr)) | (arr <= -3.0e38)

    itemsize = np.dtype(arr.dtype).itemsize
    markers = (0, 255) if itemsize == 1 else (0,)
    mask = np.zeros(arr.shape, dtype=bool)
    for value in markers:
        mask |= arr == value
    return mask


def compute_invalid_mask(array):
    """Mark NULL / saturation pixels (dtype-aware, see :func:`_pixel_invalid`).

    A pixel is invalid when it is invalid across every channel.
    """
    np = require_numpy()

    arr = np.asarray(array)
    if arr.ndim == 2:
        invalid_mask = _pixel_invalid(arr)
    elif arr.ndim == 3:
        invalid_mask = np.all(_pixel_invalid(arr), axis=2)
    else:
        raise ValueError(f"Unexpected patch shape: {arr.shape}")

    return invalid_mask


def _normalize_image_array(array):
    np = require_numpy()

    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D image array, got shape {arr.shape}."
        )

    if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
        return np.moveaxis(arr, 0, -1)

    return arr


def open_image_reader(path: Path) -> BaseImageReader:
    errors: list[str] = []

    try:
        return RasterioImageReader(path)
    except Exception as exc:
        errors.append(f"rasterio: {exc}")

    np = require_numpy()

    try:
        import glymur

        return ArrayImageReader(path, glymur.Jp2k(str(path))[:], "glymur")
    except Exception as exc:
        errors.append(f"glymur: {exc}")

    try:
        from PIL import Image

        with Image.open(path) as image:
            return ArrayImageReader(path, np.asarray(image), "pillow")
    except Exception as exc:
        errors.append(f"Pillow: {exc}")

    try:
        import imageio.v3 as iio

        return ArrayImageReader(path, iio.imread(path), "imageio")
    except Exception as exc:
        errors.append(f"imageio: {exc}")

    details = "\n".join(f"  - {line}" for line in errors)
    raise RuntimeError(
        f"Unable to open {path} as JP2.\n"
        "Install at least one supported reader backend.\n"
        f"{details}"
    )


def build_full_invalid_mask(
    reader: BaseImageReader,
    *,
    stripe_height: int = DEFAULT_MASK_STRIPE_HEIGHT,
):
    np = require_numpy()

    stripe_height = max(1, min(int(stripe_height), reader.height))
    invalid_mask = np.zeros((reader.height, reader.width), dtype=bool)

    for top in range(0, reader.height, stripe_height):
        height = min(stripe_height, reader.height - top)
        stripe = reader.read_region(top, 0, height, reader.width)
        invalid_mask[top : top + height, :] = compute_invalid_mask(stripe)

    return invalid_mask


def maybe_build_integral_image(black_mask):
    np = require_numpy()

    if int(np.prod(black_mask.shape, dtype=np.int64)) > MAX_INTEGRAL_IMAGE_PIXELS:
        return None

    integral = np.asarray(black_mask, dtype=np.uint32)
    integral = integral.cumsum(axis=0, dtype=np.uint32)
    integral = integral.cumsum(axis=1, dtype=np.uint32)
    return integral


def query_integral_sum(integral, top: int, left: int, height: int, width: int) -> int:
    bottom = top + height - 1
    right = left + width - 1

    total = int(integral[bottom, right])
    if top > 0:
        total -= int(integral[top - 1, right])
    if left > 0:
        total -= int(integral[bottom, left - 1])
    if top > 0 and left > 0:
        total += int(integral[top - 1, left - 1])
    return total


def extract_mask_window(
    mask,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
):
    return mask[top : top + height, left : left + width]


def compute_fraction_from_mask_window(mask_window) -> float:
    return float(mask_window.mean())


RASTER_SUFFIXES = (".jp2", ".tif", ".tiff", ".vrt")


def find_jp2_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    """Raster sources matching ``pattern``.

    Historically JP2-only (HiRISE RDR strips); GeoTIFF is accepted too so the
    NOAH-H DRG mosaic tiles can be cropped by the same path. ``pattern`` still
    decides which of those are picked up (e.g. ``*.tif``).
    """
    iterator: Iterable[Path]
    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.glob("*")

    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in RASTER_SUFFIXES
        and fnmatch.fnmatch(path.name.lower(), pattern.lower())
    ]
    return sorted(files)


def iter_start_positions(length: int, stride: int) -> list[int]:
    if length <= 0:
        return [0]
    starts = list(range(0, length, stride))
    return starts or [0]


def iter_patch_specs(
    image_height: int,
    image_width: int,
    patch_size: int,
    stride: int,
) -> Iterator[PatchSpec]:
    row_starts = iter_start_positions(image_height, stride)
    col_starts = iter_start_positions(image_width, stride)

    for row, top in enumerate(row_starts):
        for col, left in enumerate(col_starts):
            valid_height = min(patch_size, image_height - top)
            valid_width = min(patch_size, image_width - left)

            yield PatchSpec(
                row=row,
                col=col,
                top=top,
                left=left,
                valid_height=valid_height,
                valid_width=valid_width,
                pad_bottom=patch_size - valid_height,
                pad_right=patch_size - valid_width,
            )


def count_patch_candidates(
    image_height: int,
    image_width: int,
    stride: int,
) -> int:
    return len(iter_start_positions(image_height, stride)) * len(
        iter_start_positions(image_width, stride)
    )


def limit_patch_specs(
    specs: list[PatchSpec],
    *,
    max_patches_per_image: int,
) -> list[PatchSpec]:
    if max_patches_per_image < 0 or len(specs) <= max_patches_per_image:
        return specs

    selected_indices = select_example_indices(len(specs), max_patches_per_image)
    return [specs[index] for index in selected_indices]


def format_image_shape(width: int, height: int) -> str:
    return f"{width}x{height}"


def inspect_image(
    image_path: Path,
    *,
    stride: int,
    max_patches_per_image: int,
) -> ImageInventory:
    with open_image_reader(image_path) as reader:
        total_candidates = count_patch_candidates(
            reader.height,
            reader.width,
            stride,
        )
        analyzed_candidates = (
            total_candidates
            if max_patches_per_image < 0
            else min(total_candidates, max_patches_per_image)
        )
        return ImageInventory(
            image_path=image_path,
            reader_backend=reader.backend,
            height=reader.height,
            width=reader.width,
            channels=reader.channels,
            dtype=reader.dtype,
            total_candidates=total_candidates,
            analyzed_candidates=analyzed_candidates,
        )


def print_run_configuration(
    *,
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    context_size: int,
    context_output_size: int,
    stride: int,
    write_shared_context_base: bool,
    write_context_cache: bool,
    max_patches_per_image: int,
    num_processes: int,
    dry_run: bool,
) -> None:
    print(f"[run] Input directory: {input_dir}")
    print(f"[run] Output directory: {output_dir}")
    print(
        f"[run] Local patch size: {patch_size}; context: "
        f"{context_size}->{context_output_size}; stride: {stride}"
    )
    print(
        f"[run] Shared context base: {'yes' if write_shared_context_base else 'no'}; "
        f"paired context cache: {'yes' if write_context_cache else 'no'}"
    )
    print(
        "[run] Max analyzed patches per image: "
        f"{'unlimited' if max_patches_per_image < 0 else max_patches_per_image}; "
        f"worker processes: {num_processes}; mode: "
        f"{'dry-run' if dry_run else 'write'}"
    )


def print_image_inventory(images: list[ImageInventory]) -> None:
    print("[images] Source inventory:")
    for inventory in images:
        cap_note = ""
        if inventory.analyzed_candidates < inventory.total_candidates:
            cap_note = (
                f" -> analyzing {inventory.analyzed_candidates}/"
                f"{inventory.total_candidates}"
            )
        else:
            cap_note = f" -> analyzing {inventory.analyzed_candidates}"

        print(
            f"  - {inventory.image_path.name}: "
            f"{format_image_shape(inventory.width, inventory.height)}, "
            f"{inventory.channels} channel(s), {inventory.dtype}, "
            f"backend={inventory.reader_backend}{cap_note}"
        )


def terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    executor.shutdown(wait=False, cancel_futures=True)

    processes = getattr(executor, "_processes", None)
    if not processes:
        return

    for process in processes.values():
        if process.is_alive():
            process.terminate()

    for process in processes.values():
        process.join(timeout=1.0)


def drain_progress_queue(progress_queue, progress_bar) -> int:
    if progress_queue is None or progress_bar is None:
        return 0

    drained = 0
    while True:
        try:
            drained += int(progress_queue.get_nowait())
        except Empty:
            break

    if drained > 0:
        progress_bar.update(drained)
    return drained


def compute_centered_window_spec(
    image_length: int,
    center: int,
    window_size: int,
) -> tuple[int, int, int, int, int]:
    requested_start = center - window_size // 2
    requested_end = requested_start + window_size

    read_start = max(requested_start, 0)
    read_end = min(requested_end, image_length)

    pad_before = max(0, -requested_start)
    pad_after = max(0, requested_end - image_length)
    valid_length = max(0, read_end - read_start)

    return requested_start, read_start, valid_length, pad_before, pad_after


def make_context_window_spec(
    image_height: int,
    image_width: int,
    center_y: int,
    center_x: int,
    context_size: int,
) -> WindowSpec:
    (
        requested_top,
        read_top,
        valid_height,
        pad_top,
        pad_bottom,
    ) = compute_centered_window_spec(image_height, center_y, context_size)
    (
        requested_left,
        read_left,
        valid_width,
        pad_left,
        pad_right,
    ) = compute_centered_window_spec(image_width, center_x, context_size)

    return WindowSpec(
        requested_top=requested_top,
        requested_left=requested_left,
        read_top=read_top,
        read_left=read_left,
        valid_height=valid_height,
        valid_width=valid_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def pad_patch(
    array,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    pad_mode: str,
    pad_value: float,
):
    np = require_numpy()

    if array.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif array.ndim == 3:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        raise ValueError(f"Unexpected patch shape: {array.shape}")

    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return np.ascontiguousarray(array)

    if pad_mode == "constant":
        return np.pad(
            array,
            pad_width,
            mode="constant",
            constant_values=pad_value,
        )

    return np.pad(array, pad_width, mode=pad_mode)


def save_patch(output_path: Path, patch) -> None:
    np = require_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, patch)


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def to_dataset_relative(path: Path, dataset_root: Path) -> str:
    return os.path.relpath(path, dataset_root)


def context_base_filename(
    image_stem: str,
    context_size: int,
    context_output_size: int,
) -> str:
    return (
        f"{image_stem}"
        f"_context_base_c{context_size:04d}"
        f"_o{context_output_size:04d}.npy"
    )


def context_cache_filename(
    image_stem: str,
    spec: PatchSpec,
    context_output_size: int,
) -> str:
    return (
        f"{image_stem}"
        f"_context_o{context_output_size:04d}"
        f"_r{spec.row:04d}"
        f"_c{spec.col:04d}"
        f"_y{spec.top:07d}"
        f"_x{spec.left:07d}.npy"
    )


def downsample_integer_factor(array, factor: int):
    np = require_numpy()

    if factor <= 1:
        return np.ascontiguousarray(array)

    arr = np.asarray(array)
    squeeze_channel = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze_channel = True
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected array shape for downsampling: {arr.shape}")

    pad_bottom = (-arr.shape[0]) % factor
    pad_right = (-arr.shape[1]) % factor
    if pad_bottom or pad_right:
        arr = pad_patch(
            arr,
            pad_top=0,
            pad_bottom=pad_bottom,
            pad_left=0,
            pad_right=pad_right,
            pad_mode="edge",
            pad_value=0.0,
        )

    arr = arr.astype(np.float32, copy=False)
    height, width, channels = arr.shape
    arr = arr.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
        channels,
    ).mean(axis=(1, 3))

    if squeeze_channel:
        return arr[:, :, 0]
    return arr


def build_shared_context_base(reader: BaseImageReader, factor: int):
    target_height = max(1, (reader.height + factor - 1) // factor)
    target_width = max(1, (reader.width + factor - 1) // factor)

    if isinstance(reader, RasterioImageReader):
        from rasterio.enums import Resampling

        data = reader._dataset.read(
            out_shape=(reader.channels, target_height, target_width),
            resampling=Resampling.average,
        )
        return _normalize_image_array(data)

    full_image = reader.read_region(0, 0, reader.height, reader.width)
    return downsample_integer_factor(full_image, factor)


def patch_filename(image_stem: str, spec: PatchSpec) -> str:
    return (
        f"{image_stem}"
        f"_r{spec.row:04d}"
        f"_c{spec.col:04d}"
        f"_y{spec.top:07d}"
        f"_x{spec.left:07d}.npy"
    )


def context_filename(image_stem: str, spec: PatchSpec) -> str:
    return (
        f"{image_stem}"
        f"_context_r{spec.row:04d}"
        f"_c{spec.col:04d}"
        f"_y{spec.top:07d}"
        f"_x{spec.left:07d}.npy"
    )


def manifest_fieldnames() -> list[str]:
    return [
        "source_image",
        "reader_backend",
        "patch_path",
        "context_path",
        "context_base_path",
        "context_cache_path",
        "local_storage",
        "context_output_size",
        "context_scale",
        "patch_name",
        "row",
        "col",
        "top",
        "left",
        "patch_size",
        "stride",
        "valid_height",
        "valid_width",
        "pad_bottom",
        "pad_right",
        "image_height",
        "image_width",
        "channels",
        "dtype",
        "pad_mode",
        "pad_value",
        "context_size",
        "context_top",
        "context_left",
        "context_valid_height",
        "context_valid_width",
        "context_pad_top",
        "context_pad_bottom",
        "context_pad_left",
        "context_pad_right",
        "invalid_fraction",
    ]


def build_manifest_row(
    dataset_root: Path,
    image_path: Path,
    patch_path: Path,
    context_base_path: Path | None,
    context_cache_path: Path | None,
    spec: PatchSpec,
    *,
    patch_size: int,
    stride: int,
    pad_mode: str,
    pad_value: float,
    reader: BaseImageReader,
    context_size: int,
    context_output_size: int,
    context_scale: int,
    context_spec: WindowSpec,
    invalid_fraction: float,
) -> dict[str, object]:
    encoded_patch_path = to_dataset_relative(patch_path, dataset_root)
    encoded_context_base = (
        to_dataset_relative(context_base_path, dataset_root)
        if context_base_path is not None
        else ""
    )
    encoded_context_cache = (
        to_dataset_relative(context_cache_path, dataset_root)
        if context_cache_path is not None
        else ""
    )
    return {
        "source_image": to_dataset_relative(image_path, dataset_root),
        "reader_backend": reader.backend,
        "patch_path": encoded_patch_path,
        "context_path": encoded_context_cache,
        "context_base_path": encoded_context_base,
        "context_cache_path": encoded_context_cache,
        "local_storage": "patch_file",
        "context_output_size": context_output_size,
        "context_scale": context_scale,
        "patch_name": patch_path.name,
        "row": spec.row,
        "col": spec.col,
        "top": spec.top,
        "left": spec.left,
        "patch_size": patch_size,
        "stride": stride,
        "valid_height": spec.valid_height,
        "valid_width": spec.valid_width,
        "pad_bottom": spec.pad_bottom,
        "pad_right": spec.pad_right,
        "image_height": reader.height,
        "image_width": reader.width,
        "channels": reader.channels,
        "dtype": reader.dtype,
        "pad_mode": pad_mode,
        "pad_value": pad_value,
        "context_size": context_size,
        "context_top": context_spec.requested_top,
        "context_left": context_spec.requested_left,
        "context_valid_height": context_spec.valid_height,
        "context_valid_width": context_spec.valid_width,
        "context_pad_top": context_spec.pad_top,
        "context_pad_bottom": context_spec.pad_bottom,
        "context_pad_left": context_spec.pad_left,
        "context_pad_right": context_spec.pad_right,
        "invalid_fraction": invalid_fraction,
    }


def extract_patches_for_image(
    image_path: Path,
    output_dir: Path,
    *,
    patch_size: int,
    context_size: int,
    context_output_size: int,
    write_shared_context_base: bool,
    write_context_cache: bool,
    context_scale: int,
    stride: int,
    pad_mode: str,
    pad_value: float,
    num_examples: int,
    max_invalid_fraction: float,
    overwrite: bool,
    dry_run: bool,
    max_patches_per_image: int,
    show_patch_progress: bool,
    progress_queue=None,
    progress_update_interval: int = DEFAULT_PARALLEL_PROGRESS_UPDATE_INTERVAL,
    progress_flush_seconds: float = DEFAULT_PROGRESS_FLUSH_SECONDS,
) -> ImageProcessingResult:
    image_output_dir = output_dir / image_path.stem
    if not dry_run:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    with open_image_reader(image_path) as reader:
        specs = list(
            iter_patch_specs(
                image_height=reader.height,
                image_width=reader.width,
                patch_size=patch_size,
                stride=stride,
            )
        )
        specs = limit_patch_specs(
            specs,
            max_patches_per_image=max_patches_per_image,
        )
        manifest_path = image_output_dir / MANIFEST_FILENAME
        fieldnames = manifest_fieldnames()
        context_base_dir = image_output_dir / CONTEXT_BASES_DIRNAME
        context_cache_dir = image_output_dir / CONTEXT_CACHE_DIRNAME
        legacy_context_dir = image_output_dir / CONTEXTS_DIRNAME
        context_base_path = (
            context_base_dir
            / context_base_filename(
                image_path.stem,
                context_size=context_size,
                context_output_size=context_output_size,
            )
            if write_shared_context_base
            else None
        )
        queued_progress = 0
        last_progress_flush = time.monotonic()

        def note_progress() -> None:
            nonlocal queued_progress, last_progress_flush

            if progress is not None:
                progress.update(1)
            if progress_queue is not None:
                queued_progress += 1
                now = time.monotonic()
                if (
                    queued_progress >= progress_update_interval
                    or (now - last_progress_flush) >= progress_flush_seconds
                ):
                    progress_queue.put(queued_progress)
                    queued_progress = 0
                    last_progress_flush = now

        def flush_progress() -> None:
            nonlocal queued_progress, last_progress_flush

            if progress_queue is not None and queued_progress > 0:
                progress_queue.put(queued_progress)
                queued_progress = 0
                last_progress_flush = time.monotonic()

        if dry_run:
            kept = 0
            skipped_invalid = 0
            full_invalid_mask = build_full_invalid_mask(reader)
            invalid_integral = maybe_build_integral_image(full_invalid_mask)
            progress = (
                create_progress_bar(
                    total=len(specs),
                    desc=f"{image_path.name} [scan:{reader.backend}]",
                    unit="patch",
                    position=1,
                    leave=False,
                )
                if show_patch_progress
                else None
            )
            try:
                for spec in specs:
                    if invalid_integral is not None:
                        local_invalid_count = query_integral_sum(
                            invalid_integral,
                            spec.top,
                            spec.left,
                            spec.valid_height,
                            spec.valid_width,
                        )
                        invalid_fraction = (
                            local_invalid_count
                            / float(spec.valid_height * spec.valid_width)
                        )
                    else:
                        local_invalid_mask = extract_mask_window(
                            full_invalid_mask,
                            top=spec.top,
                            left=spec.left,
                            height=spec.valid_height,
                            width=spec.valid_width,
                        )
                        invalid_fraction = compute_fraction_from_mask_window(
                            local_invalid_mask
                        )

                    if (
                        max_invalid_fraction >= 0.0
                        and invalid_fraction > max_invalid_fraction
                    ):
                        skipped_invalid += 1
                    else:
                        kept += 1

                    note_progress()
            finally:
                flush_progress()
                if progress is not None:
                    progress.close()

            print(
                f"[dry-run] {image_path.name}: "
                f"{reader.height}x{reader.width}, "
                f"{len(specs)} candidates, "
                f"{kept} kept, "
                f"{skipped_invalid} skipped, "
                f"{min(num_examples, kept)} examples, "
                f"context_size={context_size} via {reader.backend}"
            )
            return ImageProcessingResult(
                image_path=str(image_path),
                manifest_path=str(manifest_path),
                kept=kept,
                skipped_invalid=skipped_invalid,
            )

        examples_dir = image_output_dir / EXAMPLES_DIRNAME
        if examples_dir.exists():
            for stale_preview in examples_dir.glob("example_*.png"):
                remove_if_exists(stale_preview)
            remove_if_exists(examples_dir / EXAMPLES_MANIFEST_FILENAME)
        if overwrite and context_cache_dir.exists():
            for stale_context in context_cache_dir.glob("*.npy"):
                remove_if_exists(stale_context)
        if overwrite and legacy_context_dir.exists():
            for stale_context in legacy_context_dir.glob("*.npy"):
                remove_if_exists(stale_context)
        if overwrite and context_base_dir.exists():
            for stale_base in context_base_dir.glob("*.npy"):
                remove_if_exists(stale_base)

        if write_shared_context_base and context_base_path is not None:
            if overwrite or not context_base_path.exists():
                shared_context_base = build_shared_context_base(reader, context_scale)
                save_patch(context_base_path, shared_context_base)

        full_invalid_mask = build_full_invalid_mask(reader)
        invalid_integral = maybe_build_integral_image(full_invalid_mask)

        with manifest_path.open("w", newline="") as manifest_file:
            manifest_writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            manifest_writer.writeheader()

            written = 0
            skipped_invalid = 0
            kept_rows: list[dict[str, object]] = []
            progress = create_progress_bar(
                total=len(specs),
                desc=f"{image_path.name} [{reader.backend}]",
                unit="patch",
                position=1,
                leave=False,
            )
            if not show_patch_progress:
                progress = None
            try:
                for spec in specs:
                    patch_path = image_output_dir / patch_filename(image_path.stem, spec)
                    context_cache_path = (
                        context_cache_dir
                        / context_cache_filename(
                            image_path.stem,
                            spec,
                            context_output_size=context_output_size,
                        )
                        if write_context_cache
                        else None
                    )
                    if invalid_integral is not None:
                        local_invalid_count = query_integral_sum(
                            invalid_integral,
                            spec.top,
                            spec.left,
                            spec.valid_height,
                            spec.valid_width,
                        )
                        invalid_fraction = (
                            local_invalid_count
                            / float(spec.valid_height * spec.valid_width)
                        )
                    else:
                        local_invalid_mask = extract_mask_window(
                            full_invalid_mask,
                            top=spec.top,
                            left=spec.left,
                            height=spec.valid_height,
                            width=spec.valid_width,
                        )
                        invalid_fraction = compute_fraction_from_mask_window(
                            local_invalid_mask
                        )

                    if (
                        max_invalid_fraction >= 0.0
                        and invalid_fraction > max_invalid_fraction
                    ):
                        remove_if_exists(patch_path)
                        if context_cache_path is not None:
                            remove_if_exists(context_cache_path)
                        skipped_invalid += 1
                        note_progress()
                        continue

                    center_y = spec.top + patch_size // 2
                    center_x = spec.left + patch_size // 2
                    context_spec = make_context_window_spec(
                        image_height=reader.height,
                        image_width=reader.width,
                        center_y=center_y,
                        center_x=center_x,
                        context_size=context_size,
                    )

                    patch = None
                    if overwrite or not patch_path.exists():
                        patch = reader.read_region(
                            spec.top,
                            spec.left,
                            spec.valid_height,
                            spec.valid_width,
                        )
                        patch = pad_patch(
                            patch,
                            pad_top=0,
                            pad_bottom=spec.pad_bottom,
                            pad_left=0,
                            pad_right=spec.pad_right,
                            pad_mode=pad_mode,
                            pad_value=pad_value,
                        )
                        save_patch(patch_path, patch)

                    if context_cache_path is not None and (
                        overwrite or not context_cache_path.exists()
                    ):
                        context_patch = reader.read_region(
                            context_spec.read_top,
                            context_spec.read_left,
                            context_spec.valid_height,
                            context_spec.valid_width,
                        )
                        context_patch = pad_patch(
                            context_patch,
                            pad_top=context_spec.pad_top,
                            pad_bottom=context_spec.pad_bottom,
                            pad_left=context_spec.pad_left,
                            pad_right=context_spec.pad_right,
                            pad_mode=pad_mode,
                            pad_value=pad_value,
                        )
                        if context_scale > 1:
                            context_patch = downsample_integer_factor(
                                context_patch,
                                context_scale,
                            )
                        save_patch(context_cache_path, context_patch)

                    row = build_manifest_row(
                        dataset_root=output_dir,
                        image_path=image_path,
                        patch_path=patch_path,
                        context_base_path=context_base_path,
                        context_cache_path=context_cache_path,
                        spec=spec,
                        patch_size=patch_size,
                        stride=stride,
                        pad_mode=pad_mode,
                        pad_value=pad_value,
                        reader=reader,
                        context_size=context_size,
                        context_output_size=context_output_size,
                        context_scale=context_scale,
                        context_spec=context_spec,
                        invalid_fraction=invalid_fraction,
                    )
                    manifest_writer.writerow(row)
                    kept_rows.append(row)
                    written += 1

                    note_progress()
            finally:
                flush_progress()
                if progress is not None:
                    progress.close()

        saved_examples = 0
        if kept_rows and num_examples > 0:
            examples_dir.mkdir(parents=True, exist_ok=True)
            selected_indices = select_example_indices(len(kept_rows), num_examples)
            np = require_numpy()
            with (examples_dir / EXAMPLES_MANIFEST_FILENAME).open("w", newline="") as f:
                example_manifest_writer = csv.DictWriter(
                    f,
                    fieldnames=preview_manifest_fieldnames(),
                )
                example_manifest_writer.writeheader()

                for example_index, kept_index in enumerate(selected_indices, start=1):
                    row = kept_rows[kept_index]
                    patch_path = output_dir / str(row["patch_path"])
                    preview_path = examples_dir / preview_filename(
                        image_path.stem,
                        example_index,
                        row=int(row["row"]),
                        col=int(row["col"]),
                        top=int(row["top"]),
                        left=int(row["left"]),
                    )
                    if overwrite or not preview_path.exists():
                        patch = np.load(patch_path)
                        save_preview_image(preview_path, patch)

                    example_manifest_writer.writerow(
                        {
                            "source_image": row["source_image"],
                            "reader_backend": reader.backend,
                            "preview_path": to_dataset_relative(preview_path, output_dir),
                            "patch_path": row["patch_path"],
                            "context_path": row["context_path"],
                            "context_base_path": row["context_base_path"],
                            "context_cache_path": row["context_cache_path"],
                            "example_index": example_index,
                            "row": row["row"],
                            "col": row["col"],
                            "top": row["top"],
                            "left": row["left"],
                            "valid_height": row["valid_height"],
                            "valid_width": row["valid_width"],
                            "pad_bottom": row["pad_bottom"],
                            "pad_right": row["pad_right"],
                            "image_height": row["image_height"],
                            "image_width": row["image_width"],
                            "channels": row["channels"],
                            "dtype": row["dtype"],
                            "pad_mode": row["pad_mode"],
                            "invalid_fraction": row["invalid_fraction"],
                        }
                    )
                    saved_examples += 1

        print(
            f"[ok] {image_path.name}: kept {written} patches, skipped "
            f"{skipped_invalid} invalid (0/255) patches, "
            f"{'wrote shared context base, ' if write_shared_context_base else ''}"
            f"{'wrote paired context cache, ' if write_context_cache else ''}"
            f"and saved {saved_examples} previews to {image_output_dir}"
        )
        return ImageProcessingResult(
            image_path=str(image_path),
            manifest_path=str(manifest_path),
            kept=written,
            skipped_invalid=skipped_invalid,
        )


def process_image_worker(
    *,
    image_path: str,
    output_dir: str,
    patch_size: int,
    context_size: int,
    context_output_size: int,
    write_shared_context_base: bool,
    write_context_cache: bool,
    context_scale: int,
    stride: int,
    pad_mode: str,
    pad_value: float,
    num_examples: int,
    max_invalid_fraction: float,
    overwrite: bool,
    dry_run: bool,
    max_patches_per_image: int,
    show_patch_progress: bool,
    progress_queue=None,
    progress_update_interval: int = DEFAULT_PARALLEL_PROGRESS_UPDATE_INTERVAL,
    progress_flush_seconds: float = DEFAULT_PROGRESS_FLUSH_SECONDS,
) -> ImageProcessingResult:
    return extract_patches_for_image(
        image_path=Path(image_path),
        output_dir=Path(output_dir),
        patch_size=patch_size,
        context_size=context_size,
        context_output_size=context_output_size,
        write_shared_context_base=write_shared_context_base,
        write_context_cache=write_context_cache,
        context_scale=context_scale,
        stride=stride,
        pad_mode=pad_mode,
        pad_value=pad_value,
        num_examples=num_examples,
        max_invalid_fraction=max_invalid_fraction,
        overwrite=overwrite,
        dry_run=dry_run,
        max_patches_per_image=max_patches_per_image,
        show_patch_progress=show_patch_progress,
        progress_queue=progress_queue,
        progress_update_interval=progress_update_interval,
        progress_flush_seconds=progress_flush_seconds,
    )


def write_global_index(
    output_path: Path,
    manifest_paths: Iterable[Path],
) -> None:
    fieldnames = manifest_fieldnames()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=fieldnames)
        writer.writeheader()

        for manifest_path in manifest_paths:
            with manifest_path.open(newline="") as manifest_file:
                reader = csv.DictReader(manifest_file)
                for row in reader:
                    writer.writerow(row)


def main() -> int:
    args = parse_args()

    if args.patch_size <= 0:
        print("--patch-size must be positive.", file=sys.stderr)
        return 1

    if args.context_size <= 0:
        print("--context-size must be positive.", file=sys.stderr)
        return 1

    if args.context_output_size is not None and args.context_output_size <= 0:
        print("--context-output-size must be positive.", file=sys.stderr)
        return 1

    if args.num_examples < 0:
        print("--num-examples must be non-negative.", file=sys.stderr)
        return 1

    if args.max_patches_per_image == 0:
        print(
            "--max-patches-per-image must be positive, or negative to disable.",
            file=sys.stderr,
        )
        return 1

    if args.max_invalid_fraction > 1.0:
        print(
            "--max-invalid-fraction must be <= 1.0, or negative to disable.",
            file=sys.stderr,
        )
        return 1

    stride = args.stride if args.stride is not None else args.patch_size
    if stride <= 0:
        print("--stride must be positive.", file=sys.stderr)
        return 1

    if args.num_processes <= 0:
        print("--num-processes must be positive.", file=sys.stderr)
        return 1

    context_output_size = (
        args.context_output_size
        if args.context_output_size is not None
        else min(args.patch_size, args.context_size)
    )
    if context_output_size > args.context_size:
        print(
            "--context-output-size must be <= --context-size.",
            file=sys.stderr,
        )
        return 1

    write_shared_context_base = not args.disable_shared_context_base
    if write_shared_context_base or args.write_context_cache:
        if args.context_size % context_output_size != 0:
            print(
                "--context-size must be divisible by --context-output-size "
                "when writing shared or cached precomputed contexts.",
                file=sys.stderr,
            )
            return 1
        context_scale = args.context_size // context_output_size
    else:
        context_scale = 1

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    images = find_jp2_files(input_dir, args.glob, args.recursive)
    if not images:
        print(
            f"No JP2 images found in {input_dir} with pattern {args.glob!r}.",
            file=sys.stderr,
        )
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / INDEX_FILENAME
    total_patches = 0
    total_skipped = 0

    try:
        print_run_configuration(
            input_dir=input_dir,
            output_dir=output_dir,
            patch_size=args.patch_size,
            context_size=args.context_size,
            context_output_size=context_output_size,
            stride=stride,
            write_shared_context_base=write_shared_context_base,
            write_context_cache=args.write_context_cache,
            max_patches_per_image=args.max_patches_per_image,
            num_processes=args.num_processes,
            dry_run=args.dry_run,
        )

        print(f"[scan] Inspecting {len(images)} JP2 image(s)...", flush=True)
        inventory_records = [
            inspect_image(
                image_path,
                stride=stride,
                max_patches_per_image=args.max_patches_per_image,
            )
            for image_path in images
        ]
        print_image_inventory(inventory_records)

        total_candidates = sum(
            record.total_candidates for record in inventory_records
        )
        total_analyzed = sum(
            record.analyzed_candidates for record in inventory_records
        )
        print(
            f"[scan] Total candidates: {total_candidates}; "
            f"configured to analyze: {total_analyzed}"
        )

        worker_kwargs = {
            "output_dir": str(output_dir),
            "patch_size": args.patch_size,
            "context_size": args.context_size,
            "context_output_size": context_output_size,
            "write_shared_context_base": write_shared_context_base,
            "write_context_cache": args.write_context_cache,
            "context_scale": context_scale,
            "stride": stride,
            "pad_mode": args.pad_mode,
            "pad_value": args.pad_value,
            "num_examples": args.num_examples,
            "max_invalid_fraction": args.max_invalid_fraction,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "max_patches_per_image": args.max_patches_per_image,
            "show_patch_progress": False,
            "progress_queue": None,
            "progress_update_interval": DEFAULT_PARALLEL_PROGRESS_UPDATE_INTERVAL,
        }

        results_by_image: dict[Path, ImageProcessingResult] = {}
        image_progress = create_progress_bar(
            total=len(images),
            desc=(
                "Images cached"
                if args.write_context_cache and not args.dry_run
                else "Images processed"
            ),
            unit="image",
            position=0,
            leave=True,
        )
        patch_progress = (
            create_progress_bar(
                total=total_analyzed,
                desc="Patches analyzed",
                unit="patch",
                position=1,
                leave=True,
            )
            if total_analyzed > 0
            else None
        )
        executor: ProcessPoolExecutor | None = None

        try:
            if args.num_processes == 1 or len(images) == 1:
                serial_worker_kwargs = dict(worker_kwargs)
                serial_worker_kwargs["progress_queue"] = LocalProgressReporter(
                    patch_progress
                )
                for image_path in images:
                    result = process_image_worker(
                        image_path=str(image_path),
                        **serial_worker_kwargs,
                    )
                    results_by_image[image_path] = result
                    total_patches += result.kept
                    total_skipped += result.skipped_invalid
                    if image_progress is not None:
                        image_progress.set_postfix(
                            {
                                "kept": total_patches,
                                "skipped": total_skipped,
                                "current": image_path.name,
                            }
                        )
                        image_progress.update(1)
                    if patch_progress is not None:
                        patch_progress.set_postfix(
                            {
                                "kept": total_patches,
                                "skipped": total_skipped,
                            }
                        )
            else:
                parallel_worker_kwargs = dict(worker_kwargs)
                with Manager() as manager:
                    parallel_worker_kwargs["progress_queue"] = manager.Queue()
                    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
                        future_to_image = {
                            executor.submit(
                                process_image_worker,
                                image_path=str(image_path),
                                **parallel_worker_kwargs,
                            ): image_path
                            for image_path in images
                        }
                        pending_futures = set(future_to_image)
                        while pending_futures:
                            done_futures, pending_futures = wait(
                                pending_futures,
                                timeout=0.2,
                                return_when=FIRST_COMPLETED,
                            )
                            drain_progress_queue(
                                parallel_worker_kwargs["progress_queue"],
                                patch_progress,
                            )
                            for future in done_futures:
                                image_path = future_to_image[future]
                                result = future.result()
                                results_by_image[image_path] = result
                                total_patches += result.kept
                                total_skipped += result.skipped_invalid
                                if image_progress is not None:
                                    image_progress.set_postfix(
                                        {
                                            "kept": total_patches,
                                            "skipped": total_skipped,
                                            "current": image_path.name,
                                        }
                                    )
                                    image_progress.update(1)
                                if patch_progress is not None:
                                    patch_progress.set_postfix(
                                        {
                                            "kept": total_patches,
                                            "skipped": total_skipped,
                                        }
                                    )
                        drain_progress_queue(
                            parallel_worker_kwargs["progress_queue"],
                            patch_progress,
                        )
                        if patch_progress is not None and patch_progress.n < total_analyzed:
                            patch_progress.update(total_analyzed - patch_progress.n)
        except KeyboardInterrupt:
            print("\n[interrupt] Stopping patch extraction...", file=sys.stderr)
            if executor is not None:
                try:
                    terminate_process_pool(executor)
                except Exception:
                    pass
            return 130
        finally:
            if patch_progress is not None:
                patch_progress.close()
            if image_progress is not None:
                image_progress.close()

        if args.dry_run:
            print(
                f"[dry-run] Total kept patches: {total_patches}; "
                f"total skipped patches: {total_skipped}"
            )
            return 0

        manifest_paths = [Path(results_by_image[image_path].manifest_path) for image_path in images]
        write_global_index(index_path, manifest_paths)
    except (ModuleNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        f"[done] Wrote {total_patches} kept patches across {len(images)} image(s)."
    )
    print(f"[done] Skipped {total_skipped} black-border patches.")
    print(f"[done] Global index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
