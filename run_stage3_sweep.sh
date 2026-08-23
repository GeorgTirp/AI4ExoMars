#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Local Bayesian sweep for stage-3 segmentation. 10 trials x 10 epochs, with
# hyperband early termination.
#
# Why this exists: the previous sweep was tuning almost nothing. Its parameter
# names were written against the pre-Muon-split config schema and were never
# updated when commit 9a0039e split the optimizer, so on the --use-muon path
# `learning_rate` / `weight_decay` were dead (read only in the non-muon branch)
# and `adam_beta1` / `adam_beta2` / `adam_eps` / `muon_ns_steps` did not name
# real config keys at all. Six of nine parameters were silent no-ops and
# muon_lr / nadam_lr -- the ones that actually drove training to NaN -- were
# never in the sweep. merge_wandb_config now raises on any override that does
# not name an existing config leaf, so that cannot silently recur.
#
# Everything NOT swept is set here on the command line: maybe_run_sweep() runs
# the trials in-process, so the sweep YAML's `command:` block is ignored and
# only its `parameters:` block matters.
#
# Swept (see config/stage3_segmentation_finetune_sweep.yaml):
#   muon_lr, nadam_lr, muon_weight_decay, warmup_fraction, muon_momentum,
#   nadam_beta1, nadam_beta2, grad_clip_norm, muon_scope, freeze_encoder_epochs
#
# Each trial writes its own best checkpoint via --per-run-checkpoint (the run id
# is inserted into the filename); without it every trial would overwrite the
# same file and only the last would survive.
#
# Usage:
#   ./run_stage3_sweep.sh                 # create a new sweep and run 10 trials
#   ./run_stage3_sweep.sh <sweep_id>      # join an existing sweep
# ---------------------------------------------------------------------------

AI4EXOMARS_ROOT="/home/georg/Documents/ESA/AI4ExoMars"
cd "$AI4EXOMARS_ROOT"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT="${WANDB_PROJECT:-noah-seg-stage3}"
EPOCHS="${EPOCHS:-10}"
COUNT="${COUNT:-10}"
# Half the train split, applied in memory (crop-cache indices preserved), to
# bring 10 trials x 10 epochs from ~44h down to ~12h on one RTX 3070.
# Batch size is unchanged, so the usual batch-size LR scaling does not apply;
# what changes is total steps (a trial is ~30k steps vs ~178k for a full
# 30-epoch run). The divergence ceiling is a per-step property and transfers,
# but optimal LR shades *down* with longer training -- so treat the winner as
# an upper estimate and re-verify at --train-fraction 1.0 before the long run.
# Validation is deliberately NOT subsampled: val/miou is what bayes and
# hyperband both optimize, and adding noise to it would corrupt the search.
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
LOADER_CONFIG="${LOADER_CONFIG:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_loader_DC_full.json}"

mkdir -p results/tune_stage3 outputs

if [[ $# -ge 1 ]]; then
  SWEEP_SELECTOR=(--wandb-sweep-id "$1")
  echo "Joining existing sweep: $1"
else
  SWEEP_SELECTOR=(--wandb-sweep-config config/stage3_segmentation_finetune_sweep.yaml)
  echo "Creating a new sweep in project '$PROJECT'"
fi

echo "Trials: $COUNT   Epochs/trial: $EPOCHS   train_fraction: $TRAIN_FRACTION"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

python -m vision_backend.train_stage3_segmentation_finetune \
  "${SWEEP_SELECTOR[@]}" \
  --wandb-sweep-count "$COUNT" \
  --wandb --wandb-mode online \
  --wandb-project "$PROJECT" \
  --wandb-group stage3-hparam-sweep \
  --wandb-job-type stage3-tune \
  --wandb-tags noah segmentation stage3 sweep local \
  --model-kind simmim \
  --encoder-checkpoint checkpoints/best_simim_25.pt \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  --num-workers 8 \
  --epochs "$EPOCHS" \
  --use-muon \
  --batch-size 4 \
  --accum-steps 2 \
  --compile \
  --channels-last \
  --train-fraction "$TRAIN_FRACTION" \
  --spatial-jitter-px 0 \
  --brightness-jitter 0 \
  --contrast-jitter 0 \
  --drop-path 0.1 \
  --decoder-channels 256 \
  --class-weight-scheme inverse_sqrt \
  --per-run-checkpoint \
  --save-optimizer-state \
  --checkpoint-path results/tune_stage3/stage3_sweep.pt \
  --history-path outputs/stage3_sweep_history.csv

echo
echo "Sweep finished. Per-trial best checkpoints:"
ls -la results/tune_stage3/ || true
