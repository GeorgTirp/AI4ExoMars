#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Does SimMIM pretraining help? One controlled A/B, not a sweep.
#
#   ARM=pretrained  HybridEncoder initialised from stage-1 SimMIM
#   ARM=scratch     the same HybridEncoder, randomly initialised
#
# The two arms differ in the encoder's starting weights and NOTHING else: same
# decoder head, same fixed hyperparameters, same seed, same data order, same
# 30-epoch schedule, same GPU model. In particular --freeze-encoder-epochs and
# --llrd are pinned for BOTH arms -- the stage-3 default freezes a pretrained
# encoder for 5 epochs while --random-init-encoder forces 0, which would make
# the arms differ in schedule as well as in initialisation.
#
# Precision is bf16 for BOTH arms. The first attempt ran fp16 and both arms hit
# NaN on nearly the same batch (2947 vs 2970 of epoch 2) at lr 4.5e-5: the
# HybridEncoder's ConvNeXt-V2 GRN takes an L2 norm over the whole spatial map,
# which overflows fp16. The encoder was pretrained in bf16 as well.
#
# Hyperparameters are the best completed trial of the architecture comparison
# (v3: lr 1.107e-4, wd 5.16e-5, global val mIoU 0.1805). The HybridEncoder is a
# different encoder from v0-v3 because the pretrained weights only fit it.
#
# Encoder checkpoint: stage1_simmim/last.pt -- the longest pretraining on disk,
# 1,200 steps at effective batch 256 (~307k samples, ~10 passes over the 30,675
# crops). Deliberately NOT a best.pt: "best" was picked by reconstruction loss,
# which does not measure representation quality.
#
#   ARM=pretrained condor_submit_bid 15 run_pretrain_ab.sub
#   ARM=scratch    condor_submit_bid 15 run_pretrain_ab.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

: "${ARM:?ARM is not set -- pretrained or scratch}"
ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/stage1_simmim/last.pt}"
case "$ARM" in
  pretrained)
    [ -f "$ENCODER_CKPT" ] || { echo "ERROR: $ENCODER_CKPT not found" >&2; exit 1; }
    INIT_ARGS=(--encoder-checkpoint "$ENCODER_CKPT") ;;
  scratch)
    INIT_ARGS=(--random-init-encoder) ;;
  *) echo "ERROR: ARM must be pretrained or scratch (got '$ARM')" >&2; exit 1 ;;
esac

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="$DER/seg_loader_DC_full.json"
CROP_CACHE_DIR="$DER/seg_crop_cache_full"
[ -f "$CROP_CACHE_DIR/meta.json" ] || { echo "ERROR: crop cache missing: $CROP_CACHE_DIR" >&2; exit 1; }

OUT_TAG="pretrain_ab_${ARM}"
mkdir -p "checkpoints/${OUT_TAG}" "job_outputs/pretrain_ab" results/pretrain_ab

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
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/pretrain_ab}"

echo "Host: $(hostname)"
echo "Arm: $ARM  (${INIT_ARGS[*]})"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  "${INIT_ARGS[@]}" \
  --global-base-grid 32 \
  --window-size 8 \
  --drop-path 0.0 \
  --decoder-channels 256 \
  --decoder-dropout 0.1 \
  --freeze-encoder-epochs 0 \
  --llrd 1.0 \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  --crop-cache-dir "$CROP_CACHE_DIR" \
  --num-workers 8 \
  --batch-size 4 \
  --epochs 30 \
  --learning-rate 1.107e-4 \
  --weight-decay 5.16e-5 \
  --ig-loss-weight 0.4 \
  --ema-decay 0.9999 \
  --seed 42 \
  --compile \
  --amp-dtype bf16 \
  --variant-id "hybrid_${ARM}" \
  --variant-metrics-path "results/pretrain_ab/${ARM}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --history-path "job_outputs/pretrain_ab/history_${ARM}.csv" \
  --wandb \
  --wandb-mode online \
  --wandb-project ai4exomars_pretrain_ab \
  --wandb-group pretrain-ab \
  --wandb-job-type "$ARM" \
  --wandb-tags pretrain-ab "$ARM" hybrid-encoder

echo "Arm $ARM finished cleanly at $(date -Is)."
