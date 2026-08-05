"""Step 2 -- build a virtual mosaic (.vrt) of the DRG tiles, no pixel copy.

Equivalent to ``gdalbuildvrt oxia_drg_v14.vrt drg_v14/*.tif`` but written
directly as GDAL VRT XML because the GDAL CLI is not installed in this env
(rasterio/GDAL is, and reads the VRT fine). All tiles share one CRS and pixel
size, so a single regular grid holds them exactly; per-tile pixel offsets are
snapped with round() to absorb float noise in the tile origins.
"""

from __future__ import annotations

import glob
from pathlib import Path
from xml.sax.saxutils import escape

import rasterio

from common import DRG_DIR, DRG_TILE_GLOB, DRG_VRT


def build_vrt(tile_dir: Path = DRG_DIR, out_path: Path = DRG_VRT) -> Path:
    tiles = sorted(glob.glob(str(tile_dir / DRG_TILE_GLOB)))
    if not tiles:
        raise FileNotFoundError(f"No DRG tiles found in {tile_dir}")

    metas = []
    left = bottom = float("inf")
    right = top = float("-inf")
    xres = yres = None
    crs_wkt = None
    dtype = nodata = None
    for f in tiles:
        with rasterio.open(f) as ds:
            b = ds.bounds
            metas.append((f, ds.width, ds.height, b))
            left = min(left, b.left)
            right = max(right, b.right)
            bottom = min(bottom, b.bottom)
            top = max(top, b.top)
            xres, yres = ds.res
            crs_wkt = ds.crs.to_wkt()
            dtype = ds.dtypes[0]
            nodata = ds.nodata

    mos_w = int(round((right - left) / xres))
    mos_h = int(round((top - bottom) / yres))
    gt = (left, xres, 0.0, top, 0.0, -yres)

    gdal_dtype = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
                  "float32": "Float32"}.get(dtype, "Byte")

    lines = [
        f'<VRTDataset rasterXSize="{mos_w}" rasterYSize="{mos_h}">',
        f"  <SRS>{escape(crs_wkt)}</SRS>",
        "  <GeoTransform>" + ", ".join(f"{v:.18g}" for v in gt) + "</GeoTransform>",
        f'  <VRTRasterBand dataType="{gdal_dtype}" band="1">',
        "    <ColorInterp>Gray</ColorInterp>",
    ]
    if nodata is not None:
        lines.append(f"    <NoDataValue>{nodata:.0f}</NoDataValue>")

    for f, w, h, b in metas:
        xoff = int(round((b.left - left) / xres))
        yoff = int(round((top - b.top) / yres))
        rel = Path(f).name  # tiles live beside the .vrt's tile_dir
        src_path = f"oxia_drg_v14/{rel}" if out_path.parent != tile_dir else rel
        lines += [
            "    <SimpleSource>",
            f'      <SourceFilename relativeToVRT="0">{escape(f)}</SourceFilename>',
            "      <SourceBand>1</SourceBand>",
            f'      <SourceProperties RasterXSize="{w}" RasterYSize="{h}" '
            f'DataType="{gdal_dtype}"/>',
            f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
            f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>',
        ]
        if nodata is not None:
            lines.append(f"      <NODATA>{nodata:.0f}</NODATA>")
        lines.append("    </SimpleSource>")

    lines += ["  </VRTRasterBand>", "</VRTDataset>", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    out = build_vrt()
    with rasterio.open(out) as ds:
        print(f"Wrote {out}")
        print(f"  mosaic size : {ds.width} x {ds.height}")
        print(f"  res (deg)   : {ds.res}")
        print(f"  bounds      : {ds.bounds}")
        print(f"  crs         : {ds.crs.to_proj4()}")
        # sanity read: a small window from the middle
        from rasterio.windows import Window

        cx, cy = ds.width // 2, ds.height // 2
        arr = ds.read(1, window=Window(cx, cy, 256, 256))
        print(f"  center 256^2: min={arr.min()} max={arr.max()} "
              f"nonzero={int((arr > 0).sum())}/{arr.size}")


if __name__ == "__main__":
    main()
