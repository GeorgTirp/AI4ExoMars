#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Does Muon help? One controlled run against the scratch arm of the
# pretraining A/B (SimMIM HybridEncoder, random init, global val mIoU 0.200).
#
# Identical to that arm -- init, decoder, seed, data order, bf16, compile,
# 30-epoch schedule, lr 1.107e-4, wd 5.16e-5 -- except the optimizer of the
# transformer blocks' 2-D matrices (qkv, proj, MLP of the 6 Swin S3 blocks and
# the S4 Swin + global-attention blocks: 32 matrices, 24.8M of 31.8M params).
# Those get Muon (KellerJordan SingleDeviceMuon, pinned f98f1ca, momentum
# 0.95, Nesterov, 5 Newton-Schulz steps); everything else stays on NAdamW
# exactly as in the baseline.
#
# The LR / WD transfer uses Moonlight's update-RMS matching (Liu et al. 2025,
# "Muon is Scalable for LLM Training"): the update is scaled to
# 0.2*sqrt(max(A,B)) so AdamW's tuned lr and decoupled wd carry over, per
# matrix lr 4.3e-4 .. 1.2e-3 in Muon units, per-step decay unchanged. Muon's
# own default (lr 0.02) would take ~10x larger per-weight steps on a model
# that already diverges at 2x the tuned AdamW lr.
#
#   condor_submit_bid 20 run_muon.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

ARM="${ARM:-scratch}"
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

OUT_TAG="muon_transformer_${ARM}"
mkdir -p "checkpoints/${OUT_TAG}" job_outputs/muon results/muon

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
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/muon}"

python -c "import muon" || { echo "ERROR: muon not installed in the venv" >&2; exit 1; }
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
  --use-muon \
  --muon-scope transformer \
  --muon-lr-mode match_adam \
  --muon-lr 1.107e-4 \
  --muon-weight-decay 5.16e-5 \
  --muon-momentum 0.95 \
  --ig-loss-weight 0.4 \
  --ema-decay 0.9999 \
  --seed 42 \
  --compile \
  --amp-dtype bf16 \
  --variant-id "muon_transformer_${ARM}" \
  --variant-metrics-path "results/muon/${ARM}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --history-path "job_outputs/muon/history_${ARM}.csv" \
  --wandb \
  --wandb-mode online \
  --wandb-project ai4exomars_muon \
  --wandb-group muon-transformer \
  --wandb-job-type "$ARM" \
  --wandb-tags muon "$ARM" hybrid-encoder

echo "Muon run ($ARM) finished cleanly at $(date -Is)."
