#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# SERVER_LAUNCH.md step 0c/4 -- the gate before buying a 16-job sweep.
#
# Question it answers: does this model learn AT ALL? v0's 2026-08-27 run spent
# 36 h and ~647,000 optimizer steps moving train_loss 2.359 -> 2.301 with
# val_miou pinned at 0.034227 for 47 consecutive epochs. Throughput fixes make
# that arrive faster, not go away.
#
# Deliberately different from a sweep trial:
#   * --ema-decay 0  -- the sweeps use 0.9999 and evaluate val on the EMA
#     weights. Over a short run the EMA has barely left its init, so val/miou
#     would look frozen even on a perfectly healthy model. TRAIN loss on the
#     live weights is the signal here.
#   * tiny --train-fraction -- this is an OVERFIT test. A model that cannot
#     drive train_loss down on a few hundred crops has a real bug; one that can
#     has a tuning problem, which is what the sweep is for.
#   * fixed --learning-rate, no wandb, no sweep -- a controlled diagnostic, not
#     a trial, so it burns no run_cap and leaves no junk runs in the project.
#
#   condor_submit_bid 20 smoke_overfit.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI4EXOMARS_ROOT="${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
cd "$AI4EXOMARS_ROOT"

mkdir -p job_outputs/smoke checkpoints/smoke outputs

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="${LOADER_CONFIG:-$DER/seg_loader_DC_full.json}"
CROP_CACHE_DIR="${CROP_CACHE_DIR:-$DER/seg_crop_cache_full}"

[ -f "$CROP_CACHE_DIR/meta.json" ] || {
  echo "ERROR: crop cache missing at $PWD/$CROP_CACHE_DIR -- run prep_crop_cache.sub first" >&2
  exit 1
}

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# The decoder runs at full 512x512 resolution, so activations dominate and peak
# memory scales hard with batch size -- BATCH_SIZE=16 OOM'd a 40 GB A100 at
# 38.4 GiB allocated. expandable_segments trims fragmentation; it does not buy
# headroom, so keep the batch size honest as well.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Host: $(hostname)"
echo "Crop cache: $CROP_CACHE_DIR"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

# v0 architecture (big encoder, context OFF) -- see run_variant_sweep.sh.
python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind context \
  --no-use-context \
  --random-init-encoder \
  --local-base-channels 52 \
  --context-base-channels 26 \
  --context-dim 256 \
  --decoder-channels 256 \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  --crop-cache-dir "$CROP_CACHE_DIR" \
  --num-workers "${NUM_WORKERS:-8}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-20}" \
  --train-fraction "${TRAIN_FRACTION:-0.01}" \
  --val-fraction-of-split "${VAL_FRACTION:-0.05}" \
  --learning-rate "${LR:-5e-4}" \
  --ig-loss-weight 0.4 \
  --decoder-dropout 0.1 \
  --ema-decay 0.0 \
  --llrd 1.0 \
  --checkpoint-path checkpoints/smoke/best.pt \
  --history-path outputs/smoke_history.csv \
  --variant-id "smoke_bs${BATCH_SIZE:-16}" \
  --variant-metrics-path results/smoke/smoke.jsonl \
  "$@"

echo "Smoke finished cleanly at $(date -Is)."
