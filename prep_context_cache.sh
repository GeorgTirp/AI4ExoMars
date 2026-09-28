#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# SERVER_LAUNCH.md step 1 -- build the context crop cache for V1/V3.
#
# CPU/IO-bound, no GPU: one boundless 2048px read per crop from the full mosaic,
# averaged down to 512 and stored uint8. ~14.6 GB for the 55,702-row manifest.
#
# Run as a Condor job rather than on the login node: the login node enforces a
# process/thread limit that kills numpy's BLAS autodetect (and anything else
# that forks), and an hour of heavy Lustre reads does not belong there.
#
#   condor_submit prep_context_cache.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI4EXOMARS_ROOT="${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
cd "$AI4EXOMARS_ROOT"

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
OUT_DIR="${OUT_DIR:-$DER/seg_context_cache_full}"

for f in "$DER/seg_crops_DC_full.csv" "$DER/drg_on_label_grid.tif"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $PWD/$f" >&2; exit 1; }
done

# prep_seg_context_cache imports vision_backend.seg_dataset, which imports torch,
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
echo "Started: $(date -Is)"

python -m vision_backend.prep_seg_context_cache \
  --manifest            "$DER/seg_crops_DC_full.csv" \
  --imagery             "$DER/drg_on_label_grid.tif" \
  --context-size        2048 \
  --context-output-size 512 \
  --out-dir             "$OUT_DIR" \
  --verify-samples      "${VERIFY_SAMPLES:-32}"

echo "Finished cleanly at $(date -Is)."
