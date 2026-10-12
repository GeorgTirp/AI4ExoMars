#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Recipe transfer: train the ConvNeXt-Swin v0 encoder (big, no context branch)
# with the CURRENT recipe, to separate encoder from training setup.
#
# The HybridEncoder's +0.023 lead over v0 (EXPERIMENT_LOG E1 vs E4) compared
# runs that also differed in setup (v0: fp16, NAdamW, swept LR, 30 epochs).
# Here v0 gets exactly the tuned HybridEncoder recipe: Muon on its transformer
# matrices (match_adam) + NAdamW, drop-path 0.087, no Lovász, bf16, compile,
# 16 epochs, seed 42, trial f1qeno3u's LRs / WDs. Reference: the HybridEncoder
# with the same recipe (E10 no_lovasz), val mIoU 0.2072.
#
# v0's code applies drop-path uniformly to its 6 Swin blocks (none on its
# ConvNeXt blocks); the HybridEncoder ramps it 0 -> 0.087 over all 14 blocks.
# The LRs / WDs were tuned for the HybridEncoder, so v0 may be off its optimum.
#
#   condor_submit_bid 20 run_recipe_v0.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

DROP_PATH=0.08716622036277302
SEED="${SEED:-42}"
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'api\.wandb\.ai' "${HOME}/.netrc"; then
  echo "ERROR: no wandb credentials found (wandb login, or export WANDB_API_KEY)." >&2
  exit 1
fi

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="$DER/seg_loader_DC_full.json"
CROP_CACHE_DIR="$DER/seg_crop_cache_full"
[ -f "$CROP_CACHE_DIR/meta.json" ] || { echo "ERROR: crop cache missing: $CROP_CACHE_DIR" >&2; exit 1; }

OUT_TAG="recipe_v0"
mkdir -p "checkpoints/${OUT_TAG}" job_outputs/recipe_v0 results/recipe_v0

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/recipe_v0}"
export WANDB_CONSOLE=wrap

python -c "import muon" || { echo "ERROR: muon not installed in the venv" >&2; exit 1; }
echo "Host: $(hostname)"
echo "Recipe transfer: ConvNeXt-Swin v0 (big, no context)  drop-path $DROP_PATH  seed $SEED"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind context \
  --no-use-context \
  --random-init-encoder \
  --local-base-channels 52 \
  --context-base-channels 26 \
  --context-dim 256 \
  --window-size 8 \
  --drop-path "$DROP_PATH" \
  --decoder-channels 256 \
  --decoder-dropout 0.1 \
  --freeze-encoder-epochs 0 \
  --llrd 1.0 \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  --crop-cache-dir "$CROP_CACHE_DIR" \
  --num-workers 8 \
  --batch-size 4 \
  --epochs 16 \
  --learning-rate 1.107e-4 \
  --weight-decay 5.16e-5 \
  --use-muon \
  --muon-scope transformer \
  --muon-lr-mode match_adam \
  --muon-lr 0.00017201125597463297 \
  --muon-weight-decay 0.010923917590074349 \
  --muon-momentum 0.95 \
  --nadam-lr 6.0799948094711265e-05 \
  --nadam-weight-decay 0.08314524730626249 \
  --ig-loss-weight 0.4 \
  --ema-decay 0.9999 \
  --seed "$SEED" \
  --compile \
  --amp-dtype bf16 \
  --variant-id "recipe_v0_s${SEED}" \
  --variant-metrics-path "results/recipe_v0/v0_s${SEED}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --wandb \
  --wandb-mode online \
  --wandb-project ai4exomars_recipe_transfer \
  --wandb-group recipe-transfer \
  --wandb-job-type v0 \
  --wandb-tags recipe-transfer v0 convnext-swin muon

echo "Recipe transfer v0 (seed $SEED) finished cleanly at $(date -Is)."
