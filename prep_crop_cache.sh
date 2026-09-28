#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# SERVER_LAUNCH.md step 1a -- build the padded (imagery, label) crop cache.
#
# REQUIRED by all four variants, not just the context ones. Without it the
# loader reads every crop live from the DEFLATE-tiled GeoTIFFs, and
# --spatial-jitter-px 32 puts each window off the tile grid, so up to 4 tiles
# are decompressed per crop per epoch. That is what made the 2026-08-27 launch
# cost 45 min/epoch and produce no results at all.
#
# CPU/IO-bound, no GPU: one padded 576px read from each of the two rasters per
# crop. ~37 GB for the 55,702-row manifest (two uint8 (N, 576, 576) arrays).
#
# Run as a Condor job rather than on the login node: the login node enforces a
# process/thread limit that kills numpy's BLAS autodetect (and anything else
# that forks), and an hour of heavy Lustre I/O does not belong there.
#
#   condor_submit prep_crop_cache.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI4EXOMARS_ROOT="${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
cd "$AI4EXOMARS_ROOT"

mkdir -p job_outputs/prep_crop_cache

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
OUT_DIR="${OUT_DIR:-$DER/seg_crop_cache_full}"
# Must be >= the training run's --spatial-jitter-px (32), or SegmentationCropDataset
# refuses the cache rather than silently reading a smaller jitter range.
JITTER_MARGIN="${JITTER_MARGIN:-32}"

for f in "$DER/seg_crops_DC_full.csv" "$DER/drg_on_label_grid.tif" "$DER/labels_DC_classid.tif"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $PWD/$f" >&2; exit 1; }
done

# prep_seg_crop_cache imports vision_backend.seg_dataset, which imports torch,
# so this needs the CUDA/cuDNN modules even though the build itself never uses a GPU.
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export PYTHONUNBUFFERED=1
# numpy's BLAS core autodetect is killed under the cluster's thread limits, and
# this job is single-threaded rasterio I/O anyway.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Out:  $PWD/$OUT_DIR"
echo "Jitter margin: $JITTER_MARGIN"
echo "Started: $(date -Is)"

python -m vision_backend.prep_seg_crop_cache \
  --manifest      "$DER/seg_crops_DC_full.csv" \
  --imagery       "$DER/drg_on_label_grid.tif" \
  --labels        "$DER/labels_DC_classid.tif" \
  --jitter-margin "$JITTER_MARGIN" \
  --out-dir       "$OUT_DIR"

echo "Finished cleanly at $(date -Is)."
