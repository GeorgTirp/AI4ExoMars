#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Reruns the "best-from-sweep" stage-3 config twice, isolating one change per
# run so each is a clean comparison against the plain baseline
# (checkpoints/stage3_segmentation_best_80ep.pt, best_val_miou=0.1066):
#
#   Run 1: + class-balanced loss + augmentation (spatial/photometric jitter)
#   Run 2: same as Run 1, + focal loss instead of plain CE
#
# All hyperparameters below (muon lr/momentum/weight-decay, nadam lr/betas,
# drop-path, decoder-channels, etc.) are copied verbatim from the checkpoint
# that produced stage3_segmentation_best_80ep.pt, so the only things that
# differ between these runs and that baseline are the additions named above.
#
# Note: the shown command this was built from had cwd=MarsObsLabeling, but its
# own --loader-config-path/--encoder-checkpoint/--checkpoint-path are relative
# paths that only resolve correctly from the AI4ExoMars repo root (that's
# where data/ and checkpoints/ live) -- so this script cd's there explicitly
# rather than relying on the caller's cwd, and activates AI4ExoMars's own
# venv itself so it doesn't matter what (if anything) was active before.
#
# Already-existing checkpoints this WILL overwrite if left at these names:
#   - checkpoints/stage3_segmentation_balanced_80ep.pt (val_miou=0.1092, from
#     an earlier run with the same balancing+augmentation settings but no
#     wandb tracking) -- Run 1 here reproduces that experiment with proper
#     wandb logging. Skip Run 1 (comment it out below) if you'd rather just
#     keep that result and only run the new focal-loss comparison.
# ---------------------------------------------------------------------------

AI4EXOMARS_ROOT="/home/georg/Documents/ESA/AI4ExoMars"
cd "$AI4EXOMARS_ROOT"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $(hostname)"
echo "Working directory: $PWD"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv || true

mkdir -p checkpoints outputs

# Everything identical to the baseline sweep-best config.
COMMON_ARGS=(
  --model-kind simmim
  --encoder-checkpoint checkpoints/best_simim_25.pt
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders
  --loader-config-path data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_loader_DC_aoi.json
  --num-workers 8
  --epochs 80
  --use-muon
  --batch-size 2
  --accum-steps 4
  --drop-path 0.1
  --decoder-channels 256
  --muon-lr 0.008707241085931095
  --muon-momentum 0.99
  --muon-weight-decay 0.017250584504870447
  --nadam-lr 0.00029230976233898245
  --nadam-beta1 0.9
  --nadam-beta2 0.99
  --warmup-fraction 0.1
  --freeze-encoder-epochs 0
  --wandb
  --wandb-mode online
  --wandb-project ai4exomars
)

# New: class-balanced loss (inverse_sqrt pixel-count weighting) + augmentation
# (spatial jitter, brightness/contrast jitter) on top of the existing flip-only
# augmentation. Applied to both runs below.
BALANCE_AND_AUGMENT_ARGS=(
  --class-weight-scheme inverse_sqrt
  --spatial-jitter-px 32
  --brightness-jitter 0.15
  --contrast-jitter 0.15
)

echo
echo "=================================================================="
echo "Run 1/2: class-balanced + augmented (plain CE)"
echo "=================================================================="
python -m vision_backend.train_stage3_segmentation_finetune \
  "${COMMON_ARGS[@]}" \
  "${BALANCE_AND_AUGMENT_ARGS[@]}" \
  --checkpoint-path checkpoints/stage3_segmentation_balanced.pt \
  --history-path outputs/stage3_segmentation_balanced_history.csv \
  --wandb-group noah-seg-retrain-balanced \
  --wandb-job-type stage3_segmentation \
  --wandb-tags noah segmentation stage3 balanced augmented best-from-sweep

echo
echo "=================================================================="
echo "Run 2/2: class-balanced + augmented + focal loss"
echo "=================================================================="
python -m vision_backend.train_stage3_segmentation_finetune \
  "${COMMON_ARGS[@]}" \
  "${BALANCE_AND_AUGMENT_ARGS[@]}" \
  --loss-kind focal \
  --focal-gamma 2.0 \
  --checkpoint-path checkpoints/stage3_segmentation_focal.pt \
  --history-path outputs/stage3_segmentation_focal_history.csv \
  --wandb-group noah-seg-retrain-focal \
  --wandb-job-type stage3_segmentation \
  --wandb-tags noah segmentation stage3 balanced augmented focal best-from-sweep

echo
echo "Both runs complete."
echo "  checkpoints/stage3_segmentation_balanced.pt"
echo "  checkpoints/stage3_segmentation_focal.pt"

echo
echo "=================================================================="
echo "Per-class global mIoU comparison vs. plain-CE baseline"
echo "=================================================================="
# run_segmentation_epoch's own val_miou is a sample-weighted average of
# *per-batch* mIoU (see scripts/eval_global_miou.py docstring) -- not
# comparable across runs at face value. This recomputes a proper global
# confusion-matrix mIoU for each checkpoint and, since --baseline is given,
# also prints a per-class IoU delta table so it's obvious whether focal loss
# actually moved any of the classes that were stuck near 0 IoU, rather than
# just watching the scalar mIoU move.
python scripts/eval_global_miou.py \
  checkpoints/stage3_segmentation_balanced.pt \
  checkpoints/stage3_segmentation_focal.pt \
  --baseline checkpoints/stage3_segmentation_best_80ep.pt
