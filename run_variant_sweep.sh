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
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI4EXOMARS_ROOT="${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
cd "$AI4EXOMARS_ROOT"

: "${VARIANT:?VARIANT is not set -- one of v0 v1 v2 v3}"
: "${SWEEP_ID:?SWEEP_ID is not set -- create it with 'wandb sweep --project ai4exomars config/variant_${VARIANT}_sweep.yaml'}"
: "${WANDB_API_KEY:?WANDB_API_KEY is not set -- export it before condor_submit (the .sub uses getenv = True)}"

# --- per-variant architecture ----------------------------------------------
# Widths calibrated locally so small/big = 0.696x params and ~0.66-0.70x train
# step time; decoder_channels is scaled too because the fixed-width decoder is
# ~73% of the step and width alone moved wall-clock by only 3%.
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
  LOCAL_BASE=44; CONTEXT_BASE=22; CONTEXT_DIM=217; DECODER_CH=192
fi

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="${LOADER_CONFIG:-$DER/seg_loader_DC_full.json}"
CONTEXT_CACHE_DIR="${CONTEXT_CACHE_DIR:-$DER/seg_context_cache_full}"
OUT_TAG="variant_${VARIANT}"

mkdir -p "checkpoints/${OUT_TAG}" "job_outputs/${OUT_TAG}"

for f in "$DER/drg_on_label_grid.tif" "$DER/labels_DC_classid.tif" "$LOADER_CONFIG"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $PWD/$f" >&2; exit 1; }
done

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
echo "Variant: $VARIANT  ($SIZE encoder, context ${CONTEXT_FLAG#--})"
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
  ${CONTEXT_ARGS[@]+"${CONTEXT_ARGS[@]}"} \
  --num-workers "${NUM_WORKERS:-8}" \
  --epochs "${EPOCHS:-50}" \
  --ig-loss-weight "${IG_LOSS_WEIGHT:-0.4}" \
  --decoder-dropout 0.1 \
  --ema-decay 0.9999 \
  --llrd 1.0 \
  --variant-id "$VARIANT" \
  --variant-metrics-path "results/variant_comparison/${OUT_TAG}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --wandb \
  --wandb-mode online \
  --wandb-project "${WANDB_PROJECT:-ai4exomars}" \
  --wandb-group "${WANDB_GROUP:-model-variants}" \
  --wandb-job-type "variant_${VARIANT}" \
  --wandb-tags variants "$VARIANT" "$SIZE" "context-${CONTEXT_FLAG#--no-}" \
  --wandb-sweep-id "$SWEEP_ID" \
  --wandb-sweep-count "${SWEEP_COUNT:-4}" \
  "$@"

echo "Agent finished cleanly at $(date -Is)."
