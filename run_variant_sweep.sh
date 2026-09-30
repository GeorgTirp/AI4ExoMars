#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Model-variant comparison agent (V0-V3). One script, four variants.
#
# The sweep YAMLs search ONLY {learning_rate, weight_decay}; the fixed
# architecture per variant lives here, because these sweeps run in-process via
# maybe_run_sweep and a `command:` block in the YAML would be silently ignored
# (same reason stage3_segmentation_finetune_sweep.yaml has none).
#
#   VARIANT=v0  big   encoder, context OFF   <- reference, no cache needed
#   VARIANT=v1  big   encoder, context ON    <- NEEDS the context cache
#   VARIANT=v2  small encoder, context OFF   <- no cache needed
#   VARIANT=v3  small encoder, context ON    <- NEEDS the context cache
#
# Usage (see vision_backend/training/SERVER_LAUNCH.md):
#   export WANDB_API_KEY=...
#   VARIANT=v0 SWEEP_ID=entity/ai4exomars/<id> ./run_variant_sweep.sh
#   VARIANT=v0 SWEEP_ID=... condor_submit run_variant_sweep.sub
#
# Throughput and memory (all measured on A100-40GB):
#   * the padded crop cache is REQUIRED -- see the CROP_CACHE_DIR check below
#   * BATCH_SIZE defaults to 4 at 512 px: batch 16 OOMs a 40 GB card and the big
#     variants already peak at ~36.5 GB at batch 4. For 1024 px inputs use
#     BATCH_SIZE=1 -- the same pixels per step, so the swept LR range still
#     applies. The swept range is only comparable across runs at equal pixels
#     per step.
#   * --compile is on (COMPILE=1): 0.45 -> 0.15 s/step for the ConvNeXt-Swin
#     variants. The HybridEncoder's GRN is excluded from compilation separately
#     (blocks_v2.GRN), because compiled GRN yields non-finite gradients.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI4EXOMARS_ROOT="${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
cd "$AI4EXOMARS_ROOT"

: "${VARIANT:?VARIANT is not set -- one of v0 v1 v2 v3}"
: "${SWEEP_ID:?SWEEP_ID is not set -- create it with 'wandb sweep --project ai4exomars config/variant_${VARIANT}_sweep.yaml'}"
# Credentials come from WANDB_API_KEY (inherited via `getenv = True`) or from
# ~/.netrc, which the execute nodes see through the shared home. The netrc is
# preferable: `getenv = True` copies the submitting environment into the job
# ClassAd, so an exported key is readable by anyone who can run `condor_q -l`.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'api\.wandb\.ai' "${HOME}/.netrc"; then
  echo "ERROR: no wandb credentials found." >&2
  echo "  Either: wandb login          (writes ~/.netrc, nothing lands in the job ad)" >&2
  echo "  Or:     export WANDB_API_KEY=...   before condor_submit_bid" >&2
  exit 1
fi

# --- per-variant architecture ----------------------------------------------
# Widths calibrated ON AN A100 (SERVER_LAUNCH.md step 2b) so small/big lands in
# the 0.68-0.72 band on BOTH ratios: 0.684x params, 0.718x train step time.
# decoder_channels is scaled too because the fixed-width decoder is ~73% of the
# step and width alone moved wall-clock by only 3%. dch=192 was the pre-launch
# guess and measured 0.758-0.762x step on two A100s -- out of band -- so the
# grid over (local_base_channels x decoder_channels) picked 176, the only point
# with both ratios inside 0.68-0.72.
case "$VARIANT" in
  v0) SIZE=big;   CONTEXT_FLAG="--no-use-context" ;;
  v1) SIZE=big;   CONTEXT_FLAG="--use-context" ;;
  v2) SIZE=small; CONTEXT_FLAG="--no-use-context" ;;
  v3) SIZE=small; CONTEXT_FLAG="--use-context" ;;
  *)  echo "ERROR: VARIANT must be one of v0 v1 v2 v3 (got '$VARIANT')" >&2; exit 1 ;;
esac

if [ "$SIZE" = "big" ]; then
  LOCAL_BASE=52; CONTEXT_BASE=26; CONTEXT_DIM=256; DECODER_CH=256
else
  LOCAL_BASE=44; CONTEXT_BASE=22; CONTEXT_DIM=217; DECODER_CH=176
fi

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="${LOADER_CONFIG:-$DER/seg_loader_DC_full.json}"
CONTEXT_CACHE_DIR="${CONTEXT_CACHE_DIR:-$DER/seg_context_cache_full}"
CROP_CACHE_DIR="${CROP_CACHE_DIR:-$DER/seg_crop_cache_full}"
# Context fusion (see ContextAwareConvNeXtSwinEncoder): "film" is the original
# pooled-vector conditioning; "xattn" keeps the context grid and cross-attends to
# it at the bottleneck. Only meaningful for the context variants, and an xattn run
# gets its own output tag + variant id so it can never be read as, or averaged
# into, the FiLM result for the same variant.
CONTEXT_FUSION="${CONTEXT_FUSION:-film}"
case "$CONTEXT_FUSION" in
  film|xattn) ;;
  *) echo "ERROR: CONTEXT_FUSION must be film or xattn (got '$CONTEXT_FUSION')" >&2; exit 1 ;;
esac
if [ "$CONTEXT_FUSION" != "film" ] && [ "$CONTEXT_FLAG" != "--use-context" ]; then
  echo "ERROR: CONTEXT_FUSION=$CONTEXT_FUSION needs a context variant (v1 or v3), got $VARIANT" >&2
  exit 1
fi
VARIANT_ID="$VARIANT"
if [ "$CONTEXT_FUSION" != "film" ]; then
  VARIANT_ID="${VARIANT}_${CONTEXT_FUSION}"
fi
# RUN_TAG separates runs that share a variant but change the data or input
# size -- e.g. RUN_TAG=1024 for the 1024x1024 single-branch test, which would
# otherwise append to variant_v2.jsonl and checkpoints/variant_v2 alongside the
# 512 results it is being compared against.
if [ -n "${RUN_TAG:-}" ]; then
  VARIANT_ID="${VARIANT_ID}_${RUN_TAG}"
fi
OUT_TAG="variant_${VARIANT_ID}"

mkdir -p "checkpoints/${OUT_TAG}" "job_outputs/${OUT_TAG}"

for f in "$DER/drg_on_label_grid.tif" "$DER/labels_DC_classid.tif" "$LOADER_CONFIG"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $PWD/$f" >&2; exit 1; }
done

# EVERY variant needs the padded crop cache. Without it the loader re-reads each
# crop from the DEFLATE-tiled GeoTIFFs, and --spatial-jitter-px puts the window
# off the tile grid so up to 4 tiles are decompressed per crop per epoch. That is
# what the first launch did: v0 spent 36 h to reach epoch 48 of trial 1 of 4 and
# was held by MaxTime having written no result at all. Fail loudly, like the
# context cache below. CROP_CACHE_OPTIONAL=1 opts out for a smoke run.
if [ ! -f "$CROP_CACHE_DIR/meta.json" ] && [ -z "${CROP_CACHE_OPTIONAL:-}" ]; then
  echo "ERROR: padded crop cache not found at:" >&2
  echo "  $PWD/$CROP_CACHE_DIR" >&2
  echo "Build it first (one pass over the mosaic, ~37 GB, I/O bound):" >&2
  echo "  python -m vision_backend.prep_seg_crop_cache \\" >&2
  echo "    --manifest $DER/seg_crops_DC_full.csv \\" >&2
  echo "    --imagery  $DER/drg_on_label_grid.tif \\" >&2
  echo "    --labels   $DER/labels_DC_classid.tif \\" >&2
  echo "    --jitter-margin 32 \\" >&2
  echo "    --out-dir  $CROP_CACHE_DIR" >&2
  echo "Or set CROP_CACHE_OPTIONAL=1 to run live-read anyway (much slower)." >&2
  exit 1
fi
CROP_CACHE_ARGS=()
if [ -f "$CROP_CACHE_DIR/meta.json" ]; then
  CROP_CACHE_ARGS=(--crop-cache-dir "$CROP_CACHE_DIR")
fi

# torch.compile. Measured 0.45 s/step uncompiled at batch 8, i.e. ~50 min/epoch
# and ~42 h for 50 epochs -- past MaxTime. The repo measured 2.1x from compile,
# which brings a full trial comfortably inside the wall. COMPILE=0 disables it
# if a backend problem shows up; the run is correct either way, just slower.
COMPILE_ARGS=()
if [ "${COMPILE:-1}" = "1" ]; then
  COMPILE_ARGS=(--compile)
fi

# Context variants are useless without the cache: the loader would silently fall
# back to a live per-item 2048px read and the run would crawl. Fail loudly.
CONTEXT_ARGS=()
if [ "$CONTEXT_FLAG" = "--use-context" ]; then
  if [ ! -f "$CONTEXT_CACHE_DIR/context.npy" ]; then
    echo "ERROR: $VARIANT needs the context crop cache, not found at:" >&2
    echo "  $PWD/$CONTEXT_CACHE_DIR/context.npy" >&2
    echo "Build it first (SERVER_LAUNCH.md step 1):" >&2
    echo "  python -m vision_backend.prep_seg_context_cache \\" >&2
    echo "    --manifest $DER/seg_crops_DC_full.csv \\" >&2
    echo "    --imagery  $DER/drg_on_label_grid.tif \\" >&2
    echo "    --context-size 2048 --context-output-size 512 \\" >&2
    echo "    --out-dir  $CONTEXT_CACHE_DIR" >&2
    exit 1
  fi
  CONTEXT_ARGS=(--context-cache-dir "$CONTEXT_CACHE_DIR")
fi

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_SILENT=false
export WANDB_CONSOLE=wrap
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/${OUT_TAG}}"

echo "Host: $(hostname)"
echo "Variant: $VARIANT_ID  ($SIZE encoder, context ${CONTEXT_FLAG#--}, fusion $CONTEXT_FUSION)"
echo "  local_base=$LOCAL_BASE context_base=$CONTEXT_BASE context_dim=$CONTEXT_DIM decoder_channels=$DECODER_CH"
echo "Sweep: $SWEEP_ID"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python - <<'PY'
import sys, wandb
try:
    print("wandb auth OK as:", wandb.Api().viewer.entity)
except Exception as exc:
    sys.exit(f"wandb authentication FAILED: {type(exc).__name__}: {exc}")
PY

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind context \
  $CONTEXT_FLAG \
  --random-init-encoder \
  --local-base-channels "$LOCAL_BASE" \
  --context-base-channels "$CONTEXT_BASE" \
  --context-dim "$CONTEXT_DIM" \
  --decoder-channels "$DECODER_CH" \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  ${CROP_CACHE_ARGS[@]+"${CROP_CACHE_ARGS[@]}"} \
  ${COMPILE_ARGS[@]+"${COMPILE_ARGS[@]}"} \
  ${CONTEXT_ARGS[@]+"${CONTEXT_ARGS[@]}"} \
  --num-workers "${NUM_WORKERS:-8}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --epochs "${EPOCHS:-50}" \
  --ig-loss-weight "${IG_LOSS_WEIGHT:-0.4}" \
  --decoder-dropout 0.1 \
  --ema-decay 0.9999 \
  --llrd 1.0 \
  --per-run-checkpoint \
  --context-fusion "$CONTEXT_FUSION" \
  --variant-id "$VARIANT_ID" \
  --variant-metrics-path "results/variant_comparison/${OUT_TAG}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --wandb \
  --wandb-mode online \
  --wandb-project "${WANDB_PROJECT:-ai4exomars}" \
  --wandb-group "${WANDB_GROUP:-model-variants}" \
  --wandb-job-type "variant_${VARIANT}" \
  --wandb-tags variants "$VARIANT" "$SIZE" "context-${CONTEXT_FLAG#--no-}" "fusion-${CONTEXT_FUSION}" \
  --wandb-sweep-id "$SWEEP_ID" \
  --wandb-sweep-count "${SWEEP_COUNT:-4}" \
  "$@"

echo "Agent finished cleanly at $(date -Is)."
