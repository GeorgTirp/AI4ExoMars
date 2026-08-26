#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Builds the two offline calibration artifacts MarsObsLabeling's mars-inference
# needs before its "Uncertainty Heatmap" button and the Class Summary window's
# Neural-PCA gallery do anything:
#
#   <checkpoint_stem>.uncertainty.pt  -> Uncertainty Heatmap button
#   <checkpoint_stem>.npca.pt         -> Neural PCA gallery (Summary window)
#
# Both are looked up as sidecars sitting NEXT TO the checkpoint (MarsObsLabeling
# inference/modelio.py:sidecar_path), and both fit scripts default --output to
# exactly that path -- so just passing --checkpoint puts each file where the GUI
# already looks. Neither existed yet, which is why both features were dead.
#
# Uses the prebuilt padded-crop cache (18GB memmap) rather than re-decompressing
# windows out of the 82-gigapixel mosaic; both scripts gained --cache-dir for
# this. --max-crops bounds the scan: fit_gaussians' own early stop only fires
# once EVERY class has filled its sample quota, and "Boulder fields" has zero
# pixels anywhere in this label raster, so without a cap it would scan all
# 47,392 train crops (one model forward each).
#
# Usage:
#   ./fit_calibration_artifacts.sh                       # default checkpoint below
#   CHECKPOINT=checkpoints/other.pt ./fit_calibration_artifacts.sh
#   MAX_CROPS=0 ./fit_calibration_artifacts.sh           # 0 = no cap, full split
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI4EXOMARS_ROOT="${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
cd "$AI4EXOMARS_ROOT"

# Activate the venv if there is one and we are not already inside it.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

CHECKPOINT="${CHECKPOINT:-checkpoints/stage3_segmentation_full_verify_50ep.pt}"
DERIVED="data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived"
MANIFEST="${MANIFEST:-$DERIVED/seg_crops_DC_full.csv}"
IMAGERY="${IMAGERY:-$DERIVED/drg_on_label_grid.tif}"
LABELS="${LABELS:-$DERIVED/labels_DC_classid.tif}"
CACHE_DIR="${CACHE_DIR:-$DERIVED/seg_crop_cache_full}"
SPLIT="${SPLIT:-train}"
MAX_CROPS="${MAX_CROPS:-4000}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

for required in "$MANIFEST" "$IMAGERY" "$LABELS"; do
  if [[ ! -f "$required" ]]; then
    echo "Required input not found: $required" >&2
    if [[ "$required" == "$MANIFEST" ]]; then
      echo >&2
      echo "Build it (deterministic -- the val split is the rightmost" >&2
      echo "--val-fraction of columns, no RNG, so this reproduces the" >&2
      echo "manifest the model was trained against):" >&2
      echo >&2
      echo "  python -m vision_backend.prep_seg_crops \\" >&2
      echo "    --imagery $IMAGERY \\" >&2
      echo "    --labels  $LABELS \\" >&2
      echo "    --crop-size 512 --min-label-frac 0.5 --val-fraction 0.15 \\" >&2
      echo "    --out $MANIFEST" >&2
    fi
    exit 1
  fi
done

# The padded-crop cache is optional: it only avoids re-decompressing mosaic
# windows. Without it both scripts still run, just slower.
CACHE_ARGS=()
if [[ -d "$CACHE_DIR" ]]; then
  CACHE_ARGS=(--cache-dir "$CACHE_DIR")
else
  echo "No crop cache at $CACHE_DIR -- reading windows from the mosaic (slower)."
fi

MAX_CROPS_ARGS=()
if [[ "$MAX_CROPS" != "0" ]]; then
  MAX_CROPS_ARGS=(--max-crops "$MAX_CROPS")
fi

echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT   max-crops: ${MAX_CROPS} (0 = uncapped)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
fi

echo
echo "=================================================================="
echo "1/2: Mahalanobis per-class Gaussians  -> Uncertainty Heatmap button"
echo "=================================================================="
python vision_backend/uncertainty/fit_gaussians.py \
  --checkpoint "$CHECKPOINT" \
  --manifest-path "$MANIFEST" \
  --imagery-path "$IMAGERY" \
  --label-path "$LABELS" \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
  --split "$SPLIT" \
  ${MAX_CROPS_ARGS[@]+"${MAX_CROPS_ARGS[@]}"}

echo
echo "=================================================================="
echo "2/2: Neural-PCA gallery  -> Summary window thumbnails"
echo "=================================================================="
# --imagery-path must match what the checkpoint was TRAINED on: the gallery
# stores each thumbnail's provenance as "<imagery stem>_<col>_<row>", and
# MarsObsLabeling's click-to-jump resolves that stem against the imagery named
# in the checkpoint's own saved loader config. A different mosaic here (e.g.
# the AOI one) still renders thumbnails, but clicking them can't navigate.
python vision_backend/pc_align/fit_neural_pca.py \
  --checkpoint "$CHECKPOINT" \
  --manifest-path "$MANIFEST" \
  --imagery-path "$IMAGERY" \
  --label-path "$LABELS" \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
  --split "$SPLIT" \
  ${MAX_CROPS_ARGS[@]+"${MAX_CROPS_ARGS[@]}"}

echo
echo "Done. Artifacts next to the checkpoint:"
ls -la "${CHECKPOINT%.pt}".uncertainty.pt "${CHECKPOINT%.pt}".npca.pt 2>&1 || true
echo
echo "Restart mars-inference with the SAME checkpoint to pick them up."
