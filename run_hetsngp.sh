#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# HetSNGP on the SimMIM HybridEncoder segmentation model: one 30-epoch run.
#
# Identical to the winning pretraining-A/B arm in every respect -- encoder
# initialisation, decoder, fixed hyperparameters (lr 1.107e-4, wd 5.16e-5),
# seed, bf16, compile, data, 30-epoch schedule -- except that decoder.head is
# the HetSNGP output layer (vision_backend/model/hetsngp.py; Fortuin et al.,
# TMLR 2022), so the A/B arm is its direct baseline.
#
# Head settings follow the paper: m = 1024 random Fourier features, kernel
# scale 1, LayerNorm'd GP input, rank-6 heteroscedastic covariance (its CIFAR
# setting; rank 7 saturated in its ablation), softmax temperature 1.0 (ablation:
# ~1 works well), no spectral normalization (dropped for its ViT backbone,
# Sec. 5.4). MC samples: 32 per pixel in training (segmentation practice:
# 20 in Monteiro et al. 2020, 50 in Kendall & Gal 2017), 256 at evaluation
# (paper: no gain beyond 100).
#
# After training, the same job fits the Laplace covariance (Eq. 5) on the
# saved EMA weights over the training split and writes it into the checkpoint.
#
#   INIT=scratch    condor_submit_bid 20 run_hetsngp.sub
#   INIT=pretrained condor_submit_bid 20 run_hetsngp.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

: "${INIT:?INIT is not set -- pretrained or scratch (the winning A/B arm)}"
ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/stage1_simmim/last.pt}"
case "$INIT" in
  pretrained)
    [ -f "$ENCODER_CKPT" ] || { echo "ERROR: $ENCODER_CKPT not found" >&2; exit 1; }
    INIT_ARGS=(--encoder-checkpoint "$ENCODER_CKPT") ;;
  scratch)
    INIT_ARGS=(--random-init-encoder) ;;
  *) echo "ERROR: INIT must be pretrained or scratch (got '$INIT')" >&2; exit 1 ;;
esac

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="$DER/seg_loader_DC_full.json"
CROP_CACHE_DIR="$DER/seg_crop_cache_full"
[ -f "$CROP_CACHE_DIR/meta.json" ] || { echo "ERROR: crop cache missing: $CROP_CACHE_DIR" >&2; exit 1; }

OUT_TAG="hetsngp_${INIT}"
mkdir -p "checkpoints/${OUT_TAG}" job_outputs/hetsngp results/hetsngp

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
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/hetsngp}"

echo "Host: $(hostname)"
echo "Init: $INIT  (${INIT_ARGS[*]})"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  "${INIT_ARGS[@]}" \
  --global-base-grid 32 \
  --window-size 8 \
  --drop-path 0.0 \
  --decoder-channels 256 \
  --decoder-dropout 0.1 \
  --uncertainty-head hetsngp \
  --gp-num-inducing 1024 \
  --gp-kernel-scale 1.0 \
  --het-num-factors 6 \
  --het-temperature 1.0 \
  --het-train-mc-samples 32 \
  --het-test-mc-samples 256 \
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
  --variant-id "hetsngp_${INIT}" \
  --variant-metrics-path "results/hetsngp/${INIT}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --history-path "job_outputs/hetsngp/history_${INIT}.csv" \
  --wandb \
  --wandb-mode online \
  --wandb-project ai4exomars_hetsngp \
  --wandb-group hetsngp \
  --wandb-job-type "$INIT" \
  --wandb-tags hetsngp "$INIT" hybrid-encoder

CKPT="$(ls -t checkpoints/${OUT_TAG}/best_*ep.pt | head -1)"
echo "Training finished; fitting the Laplace covariance on $CKPT"
PYTHONPATH=".:${PYTHONPATH:-}" python scripts/fit_hetsngp_covariance.py "$CKPT" \
  --crop-cache-dir "$CROP_CACHE_DIR" --batch-size 16 --num-workers 8 --amp-dtype bf16

echo "HetSNGP ($INIT) finished cleanly at $(date -Is)."
