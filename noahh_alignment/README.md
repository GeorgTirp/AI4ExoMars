# HiRISE ↔ NOAH-H label alignment (Route A)

Registers the HiRISE DRG orthomosaic onto the **NOAH-H label mosaic grid** so
labels and imagery overlay pixel-for-pixel, and decodes the RGBA label mosaics
to single-band class-index rasters on that same grid. Everything is driven from
the label file — no grid/CRS constants are hardcoded.

All I/O is via **rasterio/GDAL 3.12 (Python)** because the GDAL command-line
tools are not installed in this environment. Where the brief cites a `gdalwarp`
/ `gdalbuildvrt` / `gdaladdo` command, the equivalent is implemented in Python.

## Data (in `data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/`)
- `Oxia_NOAHH_Mosaic_DC_..._lon0.tif` — Descriptive Classes, RGBA (14 classes). **present**
- `Oxia_NOAHH_Mosaic_IG_..._lon0.tif` — Interpretive Groups (5 groups). **MISSING** (only `.aux.xml`)
- `*_tile_legend.csv` — colour→class legends. present (both)
- `oxia_drg_v17/UDL_HiRISE-Mosaic_DRG_V1.7-tile-*.tif` — 20 DRG tiles, geographic
  lon/lat on the Mars-2000 **sphere** (`+proj=longlat +R=3396190`), ~0.25 m/px. **present**

The pipeline auto-discovers the highest `oxia_drg_v*` folder, so swapping DRG
versions needs no code change (`common.discover_drg`).

## The frame mismatch it corrects
| | label mosaic (target) | DRG tiles (source) |
|---|---|---|
| CRS | eqc **metres**, lon_0=0 | geographic **degrees**, native |
| figure | Mars-2000 **ellipsoid** (rf=169.894) | Mars-2000 **sphere** (R=3396190) |
| pixel | 0.241384614 m | ~0.25 m (4.2396e-6°) |

Both CRSs are fully defined, so PROJ applies an explicit transform (no null
datum shift). Latitude is carried through as-is (sphere longlat → ellipsoid
eqc); the alignment check confirms this introduces **no** planetocentric/
planetographic offset (that would be ~1 km ≈ 4000 px and unmissable).

## Pipeline
```bash
cd noahh_alignment
python step1_target_grid.py         # read label grid + legends -> derived/target_grid.json
python build_vrt.py                 # virtual DRG mosaic       -> derived/oxia_drg_v17.vrt
python warp_drg.py  [--center-lonlat LON LAT --size N | --aoi XMIN YMIN XMAX YMAX]
python decode_labels.py --kind DC   # RGBA -> class ids        -> derived/labels_DC_classid.tif
python verify_alignment.py          # residual + overlays      -> derived/alignment_report.md
python package.py                   # internal overviews (COG-style)
```

### Full-scene warp (whole DRG extent onto the label grid)
`warp_drg.py` defaults to a verification AOI. For the entire mosaic, derive the
DRG union bounds in target-CRS metres from the VRT (don't hand-type them):
```bash
python - <<'PY'
import rasterio, math
from common import DRG_VRT
a=3396190.0; d=a*math.pi/180
with rasterio.open(DRG_VRT) as v: b=v.bounds
print(f"--aoi {b.left*d:.1f} {b.bottom*d:.1f} {b.right*d:.1f} {b.top*d:.1f}")
PY
# -> --aoi -1464077.8 1055083.6 -1416657.5 1096576.2   (V1.7)
python warp_drg.py --aoi -1464077.8 1055083.6 -1416657.5 1096576.2 \
  --out ../data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/drg_on_label_grid.tif
python package.py ../data/.../derived/drg_on_label_grid.tif   # overviews
```
The full output is ~196000×172000 px (single band) — tens of GB even DEFLATE-
compressed; make sure there is disk headroom. Equivalent `gdalwarp` (if the CLI
is installed), recorded per the brief:
```
gdalwarp -s_srs "+proj=longlat +R=3396190 +no_defs" \
  -t_srs "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=3396190 +rf=169.894447223612 +units=m +no_defs" \
  -tr 0.241384614 0.241384614 -tap -te -1464077.8 1055083.6 -1416657.5 1096576.2 \
  -r cubic -multi -co TILED=YES -co COMPRESS=DEFLATE -co BIGTIFF=YES \
  derived/oxia_drg_v17.vrt derived/drg_on_label_grid.tif
```

## Deliverables (in `derived/`)
- `target_grid.json` — the exact target grid (CRS WKT, transform, size, bounds).
- `oxia_drg_v17.vrt` — virtual DRG mosaic (188699×165112).
- `drg_on_label_grid_aoi.tif` — DRG warped to the label grid over the verification
  AOI (same CRS, 0.241384614 m pixel, origin phase); tiled + overviews.
- `labels_DC_classid.tif` — full single-band DC class-index raster (0=nodata),
  colormap attached, `classes_DC.json` sidecar. 273374×300446, ~1.4 GB.
- `alignment_report.md`, `overlay_asis.png`, `overlay_corrected.png`.

## Result
V1.7 registers to the labels **at zero shift** — the separability and edge-energy
sweeps are flat across ±24 px (no resolvable offset; a gross latitude blunder
would swing them hard). Residual is at/below the tie-signal resolution (~1 m),
comparable to NOAH-H's own hand-digitisation. See `alignment_report.md`.

## Not done here (needs inputs not present locally)
- **`labels_IG_classid.tif`** — the IG mosaic `.tif` is missing (only its
  `.aux.xml`). The decoder is ready: `python decode_labels.py --kind IG` once the
  file is present. (IG could alternatively be derived from DC via the group
  mapping, but that mapping is not in these files, so it is not assumed.)
- **DRG version test (brief Step 5)** — moot: V1.7 is the version the Feb-2022
  labels postdate, and it verifies clean. V1.4 (tested earlier) is superseded.
- **Sub-pixel (<1 px) certification** — needs manual crater tie-points in QGIS;
  NOAH-H boundaries are too generalised to certify sub-pixel automatically.
```
