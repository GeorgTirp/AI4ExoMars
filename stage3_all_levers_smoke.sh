#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# 3-epoch smoke test: full manifest, ALL SIX new levers on simultaneously
# (IG aux head, balanced-softmax loss, LLRD, decoder dropout, EMA, ASPP),
# optimizer hyperparameters fixed at the verified winner (same values as
# stage3_verify_winner_full_data.sh, which produced the 0.267 global-mIoU
# baseline).
#
# This is a quick "does it run, does the combination move val/miou at all"
# check -- NOT an ablation. At 3 epochs (and everything on at once) you won't
# learn which lever did what; for that, use run_stage3_new_levers_sweep.sh
# (already set up, not yet launched) once you have time for the longer sweep.
#
# On-values used below are each lever's own "suggested on-value" from its
# --help text:
#   --ig-loss-weight 0.4        (F1)
#   --loss-kind balanced_softmax (F2)
#   --llrd 0.8                  (F3)
#   --decoder-dropout 0.1       (F4)
#   --ema-decay 0.9999          (F5)
#   --use-aspp                  (M1, default rates 6,12,18)
# --class-weight-scheme is explicitly "none" since balanced_softmax doesn't
# stack with class weights (run_segmentation_epoch would otherwise warn and
# drop it anyway).
# ---------------------------------------------------------------------------

AI4EXOMARS_ROOT="/home/georg/Documents/ESA/AI4ExoMars"
cd "$AI4EXOMARS_ROOT"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

mkdir -p checkpoints outputs

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  --encoder-checkpoint checkpoints/best_simim_25.pt \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_loader_DC_full.json \
  --num-workers 8 \
  --epochs 3 \
  --train-fraction 1.0 \
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
  --class-weight-scheme none \
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
  --ig-loss-weight 0.4 \
  --loss-kind balanced_softmax \
  --llrd 0.8 \
  --decoder-dropout 0.1 \
  --ema-decay 0.9999 \
  --use-aspp \
  --save-optimizer-state \
  --checkpoint-path checkpoints/stage3_all_levers_smoke.pt \
  --history-path outputs/stage3_all_levers_smoke_history.csv \
  --wandb --wandb-mode online \
  --wandb-project noah-seg-stage3 \
  --wandb-group stage3-all-levers-smoke \
  --wandb-job-type stage3-smoke \
  --wandb-tags noah segmentation stage3 all-levers full-data 3ep-smoke

echo
echo "Done. Checkpoint: checkpoints/stage3_all_levers_smoke_3ep.pt"
echo "History:          outputs/stage3_all_levers_smoke_history.csv"
