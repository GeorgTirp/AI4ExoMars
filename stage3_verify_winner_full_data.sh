#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Verifies the sweep-winning hyperparameters transfer from the sweep's
# --train-fraction 0.5 (run_stage3_sweep.sh) to the full manifest at
# --train-fraction 1.0, --epochs 1 -- exactly the check that script's own
# comment calls out as required before committing to a long run:
#   "treat the winner as an upper estimate and re-verify at
#    --train-fraction 1.0 before the long run."
#
# Everything below except muon_lr/nadam_lr/muon_scope/nadam_beta1/
# nadam_beta2/muon_momentum/grad_clip_norm/warmup_fraction/
# muon_weight_decay/freeze_encoder_epochs (the swept values, substituted in
# from the winning wandb run config) is copied verbatim from
# run_stage3_sweep.sh's fixed args, so this is an apples-to-apples check of
# the same configuration on more data for less time, not a different setup.
#
# None of the new F1-F5 performance levers (IG aux head, balanced-softmax/
# logit-adjusted loss, LLRD, decoder dropout, EMA) are enabled here on
# purpose -- this is a clean read on whether the sweep winner itself holds up
# on full data. Those levers get their own sweep axes in
# config/stage3_segmentation_finetune_sweep.yaml for separate tuning.
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
  --epochs 50 \
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
  --save-optimizer-state \
  --checkpoint-path checkpoints/stage3_segmentation_full_verify.pt \
  --history-path outputs/stage3_full_verify_history.csv \
  --wandb --wandb-mode online \
  --wandb-project noah-seg-stage3 \
  --wandb-group stage3-full-data-verify \
  --wandb-job-type stage3-verify \
  --wandb-tags noah segmentation stage3 sweep-winner full-data 1ep-smoke

echo
echo "Done. Checkpoint: checkpoints/stage3_segmentation_full_verify_1ep.pt"
echo "History:          outputs/stage3_full_verify_history.csv"
