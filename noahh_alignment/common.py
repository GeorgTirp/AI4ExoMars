"""Shared helpers for the HiRISE <-> NOAH-H label alignment pipeline.

Route A: standardise everything on the NOAH-H label mosaic grid. Nothing here
hardcodes the grid or CRS -- they are always read from the label file itself
(guardrail: read the target grid from the label file, never assume 0.25 m).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Paths (repo-relative; the label folder is the single source of truth)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MOSAIC_DIR = REPO_ROOT / "data" / "2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic"
DERIVED_DIR = MOSAIC_DIR / "derived"

DC_LABEL = MOSAIC_DIR / "Oxia_NOAHH_Mosaic_DC_4bit25cm_20220203_rgb_lon0.tif"
IG_LABEL = MOSAIC_DIR / "Oxia_NOAHH_Mosaic_IG_4bit25cm_20220202_rgb_lon0.tif"
DC_LEGEND = MOSAIC_DIR / "Oxia_NOAHH_Mosaic_DC_4bit25cm_20220203_rgb_tile_legend.csv"
IG_LEGEND = MOSAIC_DIR / "Oxia_NOAHH_Mosaic_IG_4bit25cm_20220202_rgb_tile_legend.csv"


def discover_drg() -> tuple[Path, str, str]:
    """Locate the DRG tile folder present locally and return
    (tile_dir, version_tag, tile_glob). Version-agnostic: picks the highest
    oxia_drg_v* directory so swapping V1.4 -> V1.7 needs no code edits."""
    candidates = sorted(MOSAIC_DIR.glob("oxia_drg_v*"))
    dirs = [d for d in candidates if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No oxia_drg_v* tile folder under {MOSAIC_DIR}")
    tile_dir = dirs[-1]  # highest version available
    sample = next(iter(tile_dir.glob("UDL_HiRISE-Mosaic_DRG_V*-tile-*.tif")), None)
    if sample is None:
        raise FileNotFoundError(f"No DRG tiles in {tile_dir}")
    # e.g. UDL_HiRISE-Mosaic_DRG_V1.7-tile-00.tif -> V1.7
    ver = sample.name.split("_DRG_")[1].split("-tile-")[0]
    tile_glob = f"UDL_HiRISE-Mosaic_DRG_{ver}-tile-*.tif"
    return tile_dir, ver, tile_glob


DRG_DIR, DRG_VERSION, DRG_TILE_GLOB = discover_drg()
DRG_VRT = DERIVED_DIR / f"oxia_drg_{DRG_VERSION.lower().replace('.', '')}.vrt"


# ---------------------------------------------------------------------------
# Target grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetGrid:
    """The exact grid every output must land on, read from the label mosaic."""

    width: int
    height: int
    crs_wkt: str
    crs_proj4: str
    transform: tuple  # affine as (a, b, c, d, e, f)
    res_x: float
    res_y: float
    left: float
    bottom: float
    right: float
    top: float

    @property
    def affine(self):
        from affine import Affine

        return Affine(*self.transform)

    def to_dict(self) -> dict:
        return asdict(self)


def read_target_grid(label_path: Path = DC_LABEL) -> TargetGrid:
    with rasterio.open(label_path) as ds:
        t = ds.transform
        b = ds.bounds
        return TargetGrid(
            width=ds.width,
            height=ds.height,
            crs_wkt=ds.crs.to_wkt(),
            crs_proj4=ds.crs.to_proj4(),
            transform=(t.a, t.b, t.c, t.d, t.e, t.f),
            res_x=t.a,
            res_y=-t.e,
            left=b.left,
            bottom=b.bottom,
            right=b.right,
            top=b.top,
        )


# ---------------------------------------------------------------------------
# Legend parsing + nearest-colour decode
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegendEntry:
    class_id: int  # 1-based; 0 is reserved for nodata
    name: str
    hex: str
    rgb: tuple


def _hex_to_rgb(h: str) -> tuple:
    h = h.strip().lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def parse_legend(csv_path: Path) -> list[LegendEntry]:
    """Return the real classes from a *_tile_legend.csv.

    The CSV interleaves spacer rows (shape='rect', value=' '/empty, colour
    '#1d1e20'); only shape='square' rows with a non-empty class name are real
    classes. '#000000' (Boulder fields / Other cover) IS a real class and is
    kept -- it is distinguished from nodata by the alpha band, not the colour.
    """
    entries: list[LegendEntry] = []
    cid = 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            shape = (row.get("shape") or "").strip().lower()
            name = (row.get("value") or "").strip()
            color = (row.get("color") or "").strip()
            if shape != "square" or not name:
                continue
            cid += 1
            entries.append(
                LegendEntry(class_id=cid, name=name, hex=color, rgb=_hex_to_rgb(color))
            )
    return entries


def build_palette(entries: list[LegendEntry]) -> np.ndarray:
    """(N, 3) int16 array of legend RGB, row i == entries[i]."""
    return np.asarray([e.rgb for e in entries], dtype=np.int16)


def decode_block_to_classid(
    rgb: np.ndarray,
    alpha: np.ndarray,
    palette: np.ndarray,
    *,
    tol_sq: float = 1600.0,
) -> np.ndarray:
    """Nearest-colour decode of an RGB(A) block to 1-based class ids.

    rgb   : (H, W, 3) uint8   -- the three colour bands
    alpha : (H, W)   uint8    -- alpha band (0 == nodata)
    palette: (N, 3) int16     -- legend colours (row i -> class id i+1)
    Returns (H, W) uint8, 0 == nodata / unmatched-beyond-tolerance.

    The rendered mosaic shifts pure-0 channels to ~3, so an exact match fails;
    nearest colour in squared-L2 with a small tolerance recovers the class.
    Pixels farther than sqrt(tol_sq) from every legend colour stay 0 so a bad
    decode surfaces as nodata rather than a silent wrong label.

    Decodes via *unique colours* -- the rendered mosaic holds only ~N distinct
    colours (plus small shifts), so matching the unique set and scattering back
    is O(unique*N) instead of O(pixels*N), which keeps memory bounded on the
    huge full-resolution label mosaic.
    """
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)  # (U,3),(P,)
    d = (
        (colors[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32))
        ** 2
    ).sum(axis=2)  # (U, N)
    nearest = d.argmin(axis=1)
    nearest_d = d[np.arange(d.shape[0]), nearest]
    color_to_id = (nearest + 1).astype(np.uint8)
    color_to_id[nearest_d > tol_sq] = 0  # too far from any legend colour
    class_id = color_to_id[inverse].reshape(h, w)
    class_id[alpha == 0] = 0  # alpha is the authoritative nodata mask
    return class_id


def build_color_lut(palette: np.ndarray, *, tol_sq: float = 1600.0) -> np.ndarray:
    """Precompute a 2^24 -> class-id table for O(1) per-pixel decode.

    Index is (r << 16) | (g << 8) | b. Building the whole table once turns the
    full-mosaic decode from O(pixels * classes) into a single fancy-index per
    block, making the run I/O-bound instead of CPU-bound. ~16.7 MB uint8.
    """
    pal = palette.astype(np.int32)  # (N, 3)
    n = pal.shape[0]
    lut = np.zeros(1 << 24, dtype=np.uint8)
    grid = np.arange(256, dtype=np.int32)
    dg = (grid[:, None] - pal[None, :, 1]) ** 2  # (256, N) over green
    db = (grid[:, None] - pal[None, :, 2]) ** 2  # (256, N) over blue
    for r in range(256):
        dr = (r - pal[:, 0]) ** 2  # (N,)
        # dist over (g, b, class): (256, 256, N)
        dist = dr[None, None, :] + dg[:, None, :] + db[None, :, :]
        nearest = dist.argmin(axis=2)  # (256, 256)
        nd = np.take_along_axis(dist, nearest[:, :, None], axis=2)[:, :, 0]
        cid = (nearest + 1).astype(np.uint8)
        cid[nd > tol_sq] = 0
        lut[(r << 16) + (grid[:, None] << 8) + grid[None, :]] = cid
    return lut


def decode_block_with_lut(rgb: np.ndarray, alpha: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Decode an (H, W, 3) uint8 block to class ids using a prebuilt 2^24 LUT."""
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)
    cid = lut[(r << 16) | (g << 8) | b]
    cid[alpha == 0] = 0
    return cid


def write_classes_json(entries: list[LegendEntry], out_path: Path, *, kind: str) -> None:
    payload = {
        "kind": kind,
        "nodata": 0,
        "classes": [
            {"id": e.class_id, "name": e.name, "hex": e.hex, "rgb": list(e.rgb)}
            for e in entries
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def gdal_colormap(entries: list[LegendEntry]) -> dict:
    """id -> (r, g, b, a) colormap for attaching to the class-index raster."""
    cmap = {0: (0, 0, 0, 0)}
    for e in entries:
        cmap[e.class_id] = (e.rgb[0], e.rgb[1], e.rgb[2], 255)
    return cmap
