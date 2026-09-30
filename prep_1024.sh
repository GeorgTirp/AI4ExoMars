#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Build the 1024x1024 single-branch dataset: manifest -> buffered split ->
# padded crop cache -> loader config. CPU/IO only, no GPU.
#
# Everything is written under /fast, NOT home: home is at 423/500 GiB and
# already over its 400 GiB soft quota, and this cache is ~33 GB.
#
# Same mosaic, same label raster, same valid/label filters and val fraction as
# the 512 set, so the only intended difference is the tile size. The report at
# the end compares coverage with the 512 manifest, because the 99%-valid filter
# rejects more 1024 windows (a bigger window is likelier to touch a nodata
# edge) -- a 1024 run that sees less ground must not be mistaken for a worse
# architecture.
#
#   condor_submit_bid 15 prep_1024.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
mkdir -p job_outputs/prep_1024

DER=data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived
OUT="${OUT:-/fast/gtirpitz/ai4exomars/derived_1024}"
MANIFEST="$OUT/seg_crops_DC_1024.csv"
CACHE="$OUT/seg_crop_cache_1024"
mkdir -p "$OUT"

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true
  module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "Host: $(hostname)   out: $OUT   started $(date -Is)"

echo "=== 1/4 manifest ==="
python -m vision_backend.prep_seg_crops \
  --imagery "$DER/drg_on_label_grid.tif" \
  --labels  "$DER/labels_DC_classid.tif" \
  --crop-size 1024 \
  --out "$MANIFEST"

echo "=== 2/4 buffered split ==="
# No context branch here, so the footprint to keep off the boundary is the crop
# half-width plus jitter: 512 + 32.
python scripts/rebuffer_seg_split.py --manifest "$MANIFEST" --half-width 544

echo "=== 3/4 padded crop cache ==="
python -m vision_backend.prep_seg_crop_cache \
  --manifest "$MANIFEST" \
  --imagery  "$DER/drg_on_label_grid.tif" \
  --labels   "$DER/labels_DC_classid.tif" \
  --jitter-margin 32 \
  --out-dir  "$CACHE"

echo "=== 4/4 loader config ==="
python - "$OUT" "$MANIFEST" "$DER" <<'PY'
import json, sys, pathlib
out, manifest, der = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]).resolve()
cfg = {"manifest_path": manifest,
       "imagery_path": str(der / "drg_on_label_grid.tif"),
       "label_path": str(der / "labels_DC_classid.tif"),
       "num_classes": 14, "augment": True}
pathlib.Path(out, "seg_loader_DC_1024.json").write_text(json.dumps(cfg, indent=2))
print("wrote", pathlib.Path(out, "seg_loader_DC_1024.json"))
PY

echo "=== coverage vs the 512 set ==="
python - "$MANIFEST" "$DER/seg_crops_DC_full.csv" <<'PY'
import csv, sys
from collections import Counter
def load(p):
    rows = list(csv.DictReader(open(p)))
    S = int(rows[0]["size"])
    c = Counter(r["split"] for r in rows)
    return S, c
for label, path in (("1024", sys.argv[1]), (" 512", sys.argv[2])):
    S, c = load(path)
    px = lambda n: n * S * S / 1e9
    print(f"  {label}: train {c['train']:>6,} ({px(c['train']):5.2f} Gpx)   "
          f"val {c['val']:>5,} ({px(c['val']):4.2f} Gpx)   buffer {c['buffer']:>4,}")
PY
du -sh "$CACHE"
echo "finished $(date -Is)"
