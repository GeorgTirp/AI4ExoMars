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
import csv
import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "hirise_patches"
INDEX_FILENAME = "patch_index.csv"
MANIFEST_FILENAME = "manifest.csv"
EXAMPLES_DIRNAME = "examples"
EXAMPLES_MANIFEST_FILENAME = "examples_manifest.csv"
CONTEXTS_DIRNAME = "contexts"


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
        default=1024,
        help="Square context window size in source-image pixels.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing patch files instead of skipping them.",
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
        "--max-black-fraction",
        type=float,
        default=0.05,
        help=(
            "Drop patches whose valid region contains more than this fraction "
            "of black pixels. Set to a negative value to disable filtering."
        ),
    )
    parser.add_argument(
        "--black-threshold",
        type=float,
        default=0.0,
        help=(
            "Pixels with value <= this threshold in every channel count as "
            "black / missing image area."
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


def create_progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(
        total=total,
        desc=desc,
        unit="patch",
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
        "black_fraction",
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


def compute_black_fraction(array, black_threshold: float) -> float:
    np = require_numpy()

    arr = np.asarray(array)
    if arr.ndim == 2:
        black_mask = arr <= black_threshold
    elif arr.ndim == 3:
        black_mask = np.all(arr <= black_threshold, axis=2)
    else:
        raise ValueError(f"Unexpected patch shape: {arr.shape}")

    return float(black_mask.mean())


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


def find_jp2_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.glob("*")

    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() == ".jp2"
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
        "context_size",
        "context_top",
        "context_left",
        "context_valid_height",
        "context_valid_width",
        "context_pad_top",
        "context_pad_bottom",
        "context_pad_left",
        "context_pad_right",
        "black_fraction",
    ]


def build_manifest_row(
    image_path: Path,
    patch_path: Path,
    context_path: Path,
    spec: PatchSpec,
    *,
    patch_size: int,
    stride: int,
    pad_mode: str,
    reader: BaseImageReader,
    context_size: int,
    context_spec: WindowSpec,
    black_fraction: float,
) -> dict[str, object]:
    return {
        "source_image": str(image_path),
        "reader_backend": reader.backend,
        "patch_path": str(patch_path),
        "context_path": str(context_path),
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
        "context_size": context_size,
        "context_top": context_spec.requested_top,
        "context_left": context_spec.requested_left,
        "context_valid_height": context_spec.valid_height,
        "context_valid_width": context_spec.valid_width,
        "context_pad_top": context_spec.pad_top,
        "context_pad_bottom": context_spec.pad_bottom,
        "context_pad_left": context_spec.pad_left,
        "context_pad_right": context_spec.pad_right,
        "black_fraction": black_fraction,
    }


def extract_patches_for_image(
    image_path: Path,
    output_dir: Path,
    *,
    patch_size: int,
    context_size: int,
    stride: int,
    pad_mode: str,
    pad_value: float,
    num_examples: int,
    max_black_fraction: float,
    black_threshold: float,
    overwrite: bool,
    dry_run: bool,
    global_writer: csv.DictWriter,
) -> tuple[int, int]:
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
        manifest_path = image_output_dir / MANIFEST_FILENAME
        fieldnames = manifest_fieldnames()

        if dry_run:
            kept = 0
            skipped_black = 0
            progress = create_progress_bar(
                total=len(specs),
                desc=f"{image_path.name} [scan:{reader.backend}]",
            )
            try:
                for spec in specs:
                    patch = reader.read_region(
                        spec.top,
                        spec.left,
                        spec.valid_height,
                        spec.valid_width,
                    )
                    black_fraction = compute_black_fraction(
                        patch,
                        black_threshold=black_threshold,
                    )
                    if (
                        max_black_fraction >= 0.0
                        and black_fraction > max_black_fraction
                    ):
                        skipped_black += 1
                    else:
                        kept += 1

                    if progress is not None:
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()

            print(
                f"[dry-run] {image_path.name}: "
                f"{reader.height}x{reader.width}, "
                f"{len(specs)} candidates, "
                f"{kept} kept, "
                f"{skipped_black} skipped, "
                f"{min(num_examples, kept)} examples, "
                f"context_size={context_size} via {reader.backend}"
            )
            return kept, skipped_black

        examples_dir = image_output_dir / EXAMPLES_DIRNAME
        contexts_dir = image_output_dir / CONTEXTS_DIRNAME
        if examples_dir.exists():
            for stale_preview in examples_dir.glob("example_*.png"):
                remove_if_exists(stale_preview)
            remove_if_exists(examples_dir / EXAMPLES_MANIFEST_FILENAME)
        if contexts_dir.exists():
            for stale_context in contexts_dir.glob("*.npy"):
                if overwrite:
                    remove_if_exists(stale_context)

        with manifest_path.open("w", newline="") as manifest_file:
            manifest_writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            manifest_writer.writeheader()

            written = 0
            skipped_black = 0
            kept_rows: list[dict[str, object]] = []
            progress = create_progress_bar(
                total=len(specs),
                desc=f"{image_path.name} [{reader.backend}]",
            )
            try:
                for spec in specs:
                    patch_path = image_output_dir / patch_filename(image_path.stem, spec)
                    context_path = contexts_dir / context_filename(image_path.stem, spec)
                    patch = reader.read_region(
                        spec.top,
                        spec.left,
                        spec.valid_height,
                        spec.valid_width,
                    )
                    black_fraction = compute_black_fraction(
                        patch,
                        black_threshold=black_threshold,
                    )

                    if (
                        max_black_fraction >= 0.0
                        and black_fraction > max_black_fraction
                    ):
                        remove_if_exists(patch_path)
                        remove_if_exists(context_path)
                        skipped_black += 1
                        if progress is not None:
                            progress.update(1)
                        continue

                    if overwrite or not patch_path.exists():
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

                    center_y = spec.top + patch_size // 2
                    center_x = spec.left + patch_size // 2
                    context_spec = make_context_window_spec(
                        image_height=reader.height,
                        image_width=reader.width,
                        center_y=center_y,
                        center_x=center_x,
                        context_size=context_size,
                    )

                    if overwrite or not context_path.exists():
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
                        save_patch(context_path, context_patch)

                    row = build_manifest_row(
                        image_path=image_path,
                        patch_path=patch_path,
                        context_path=context_path,
                        spec=spec,
                        patch_size=patch_size,
                        stride=stride,
                        pad_mode=pad_mode,
                        reader=reader,
                        context_size=context_size,
                        context_spec=context_spec,
                        black_fraction=black_fraction,
                    )
                    manifest_writer.writerow(row)
                    global_writer.writerow(row)
                    kept_rows.append(row)
                    written += 1

                    if progress is not None:
                        progress.update(1)
            finally:
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
                    patch_path = Path(str(row["patch_path"]))
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
                            "source_image": str(image_path),
                            "reader_backend": reader.backend,
                            "preview_path": str(preview_path),
                            "patch_path": str(patch_path),
                            "context_path": row["context_path"],
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
                            "black_fraction": row["black_fraction"],
                        }
                    )
                    saved_examples += 1

        print(
            f"[ok] {image_path.name}: kept {written} patches, skipped "
            f"{skipped_black} black-border patches, saved raw contexts "
            f"at {context_size} px, and saved "
            f"{saved_examples} previews to {image_output_dir}"
        )
        return written, skipped_black


def main() -> int:
    args = parse_args()

    if args.patch_size <= 0:
        print("--patch-size must be positive.", file=sys.stderr)
        return 1

    if args.context_size <= 0:
        print("--context-size must be positive.", file=sys.stderr)
        return 1

    if args.num_examples < 0:
        print("--num-examples must be non-negative.", file=sys.stderr)
        return 1

    if args.max_black_fraction > 1.0:
        print(
            "--max-black-fraction must be <= 1.0, or negative to disable.",
            file=sys.stderr,
        )
        return 1

    if args.black_threshold < 0.0:
        print("--black-threshold must be non-negative.", file=sys.stderr)
        return 1

    stride = args.stride if args.stride is not None else args.patch_size
    if stride <= 0:
        print("--stride must be positive.", file=sys.stderr)
        return 1

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
        if args.dry_run:
            print(
                f"[dry-run] Found {len(images)} JP2 images under {input_dir}",
                flush=True,
            )
            global_writer = csv.DictWriter(
                sys.stdout,
                fieldnames=manifest_fieldnames(),
            )
            for image_path in images:
                kept, skipped = extract_patches_for_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    patch_size=args.patch_size,
                    context_size=args.context_size,
                    stride=stride,
                    pad_mode=args.pad_mode,
                    pad_value=args.pad_value,
                    num_examples=args.num_examples,
                    max_black_fraction=args.max_black_fraction,
                    black_threshold=args.black_threshold,
                    overwrite=args.overwrite,
                    dry_run=True,
                    global_writer=global_writer,
                )
                total_patches += kept
                total_skipped += skipped
            print(
                f"[dry-run] Total kept patches: {total_patches}; "
                f"total skipped patches: {total_skipped}"
            )
            return 0

        with index_path.open("w", newline="") as index_file:
            global_writer = csv.DictWriter(
                index_file,
                fieldnames=manifest_fieldnames(),
            )
            global_writer.writeheader()

            for image_path in images:
                kept, skipped = extract_patches_for_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    patch_size=args.patch_size,
                    context_size=args.context_size,
                    stride=stride,
                    pad_mode=args.pad_mode,
                    pad_value=args.pad_value,
                    num_examples=args.num_examples,
                    max_black_fraction=args.max_black_fraction,
                    black_threshold=args.black_threshold,
                    overwrite=args.overwrite,
                    dry_run=False,
                    global_writer=global_writer,
                )
                total_patches += kept
                total_skipped += skipped
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
