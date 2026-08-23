#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Sweeps the F1-M1 performance levers (IG aux head, balanced-softmax/
# logit-adjusted loss, LLRD, decoder dropout, EMA, ASPP) on the full manifest,
# holding the optimizer hyperparameters fixed at the values that produced the
# verified 0.267 global-mIoU baseline (stage3_verify_winner_full_data.sh) --
# see config/stage3_new_levers_sweep.yaml for why this is a separate sweep
# from stage3_segmentation_finetune_sweep.yaml rather than added to it.
#
# NOT launched automatically -- run this yourself when ready:
#   ./run_stage3_new_levers_sweep.sh                 # create a new sweep
#   ./run_stage3_new_levers_sweep.sh <sweep_id>       # join an existing one
#
# Budget (overridable via env vars): 25 epochs/trial, half the train split,
# both applied in memory -- roughly a quarter of the ~15.5h the full
# 50-epoch/full-data verify run took, since that run's val/miou had already
# plateaued by epoch ~20-25. 12 trials x hyperband early-termination
# (brackets at epoch 3 and 9, both well inside the 25-epoch budget).
# ---------------------------------------------------------------------------

AI4EXOMARS_ROOT="/home/georg/Documents/ESA/AI4ExoMars"
cd "$AI4EXOMARS_ROOT"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT="${WANDB_PROJECT:-noah-seg-stage3}"
EPOCHS="${EPOCHS:-25}"
COUNT="${COUNT:-12}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
LOADER_CONFIG="${LOADER_CONFIG:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_loader_DC_full.json}"

mkdir -p results/tune_stage3_new_levers outputs

if [[ $# -ge 1 ]]; then
  SWEEP_SELECTOR=(--wandb-sweep-id "$1")
  echo "Joining existing sweep: $1"
else
  SWEEP_SELECTOR=(--wandb-sweep-config config/stage3_new_levers_sweep.yaml)
  echo "Creating a new sweep in project '$PROJECT'"
fi

echo "Trials: $COUNT   Epochs/trial: $EPOCHS   train_fraction: $TRAIN_FRACTION"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

python -m vision_backend.train_stage3_segmentation_finetune \
  "${SWEEP_SELECTOR[@]}" \
  --wandb-sweep-count "$COUNT" \
  --wandb --wandb-mode online \
  --wandb-project "$PROJECT" \
  --wandb-group stage3-new-levers-sweep \
  --wandb-job-type stage3-tune \
  --wandb-tags noah segmentation stage3 sweep new-levers full-data \
  --model-kind simmim \
  --encoder-checkpoint checkpoints/best_simim_25.pt \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  --num-workers 8 \
  --epochs "$EPOCHS" \
  --train-fraction "$TRAIN_FRACTION" \
  --use-muon \
  --batch-size 4 \
  --accum-steps 2 \
  --compile \
  --channels-last \
  --spatial-jitter-px 0 \
  --brightness-jitter 0 \
  --contrast-jitter 0 \
  --drop-path 0.1 \
  --decoder-channels 256 \
  --class-weight-scheme inverse_sqrt \
  --muon-lr 0.003124027454067084 \
  --nadam-lr 0.00003050160816782696 \
  --muon-scope matrix \
  --nadam-beta1 0.9341350239536628 \
  --nadam-beta2 0.9513643087247468 \
  --muon-momentum 0.872444947776662 \
  --grad-clip-norm 0.5 \
  --warmup-fraction 0.0639822376824684 \
  --muon-weight-decay 0.00045753765372136913 \
  --freeze-encoder-epochs 0 \
  --per-run-checkpoint \
  --save-optimizer-state \
  --checkpoint-path results/tune_stage3_new_levers/stage3_new_levers.pt \
  --history-path outputs/stage3_new_levers_history.csv

echo
echo "Sweep finished. Per-trial best checkpoints:"
ls -la results/tune_stage3_new_levers/ || true
