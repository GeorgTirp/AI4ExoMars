#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Stage-3 on the FULL manifest, first attempt after the NaN post-mortem.
#
# What killed the previous run (checkpoints/stage3_segmentation_fullcheck_30ep.pt,
# epoch 1, val_miou 0.0996 -- the only clean epoch it ever produced):
#
#   The swept hyperparameters come from the AOI manifest: 208 train crops =
#   26 optimizer steps/epoch = 2,080 steps over 80 epochs. The full manifest is
#   47,392 train crops = 5,924 steps/epoch = ~178k steps over 30 epochs, i.e.
#   ~85x more updates at the same peak LR. Warmup (10%) therefore ended around
#   epoch 3, peak muon-lr 8.7e-3 landed at epoch 4 -- exactly where train_loss
#   went nan, with val_loss already climbing (1.128 -> 1.274 -> 1.285) through
#   warmup beforehand. Nothing bounded the update and nothing aborted the run,
#   so it burned ~8 further hours emitting nan.
#
# Changes here, deliberately kept to the minimum needed to test that diagnosis:
#   1. muon-lr 8.707e-3 -> 1.741e-3 (5x damped). nadam-lr is left alone; at
#      2.9e-4 it was never the aggressive one.
#   2. --grad-clip-norm 1.0  (there was previously NO clipping anywhere).
#   3. non-finite loss now aborts by default -- no more silent nan epochs.
#   4. --muon-scope matrix. The old routing reused the *weight-decay* predicate
#      (">=2-D") to decide *which optimizer* a tensor goes to, so 48 of the 80
#      tensors Muon was driving were convolutions -- including depthwise
#      [C,1,7,7] filters, which Muon flattens to [C,49] and orthogonalises,
#      forcing mutually-orthogonal filters across channels that are independent
#      by construction. Muon is validated on transformer weight matrices; now
#      only the 32 genuine 2-D attention/MLP weights go to it and the convs go
#      to NAdam (weight decay preserved). Set --muon-scope all to A/B it.
#
# Also new since that run (not hypotheses -- measured / verified here):
#   * val split is now a random draw (scripts/resplit_seg_manifest.py). The old
#     contiguous right-hand band gave only 3.1% val and up to 8.3x class skew
#     between train and val; it is now 14.9% val with worst skew 1.18x.
#     NOTE: a random split leaks spatially (neighbouring 512px tiles land on
#     both sides), so treat this val number as optimistic -- good for "is it
#     learning", not for a number you report. For that, re-split with
#     `--mode blocks` (worst skew 1.34x, val terrain spatially separated).
#   * --compile + --channels-last, and the decoder now classifies before
#     upsampling instead of after (mathematically identical, verified to 4e-15).
#     Together: 13.7 -> 39.9 img/s on this RTX 3070, ~65 -> ~22 min/epoch.
#   * --save-optimizer-state so this run is actually resumable via --resume-from.
# ---------------------------------------------------------------------------

AI4EXOMARS_ROOT="/home/georg/Documents/ESA/AI4ExoMars"
cd "$AI4EXOMARS_ROOT"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $(hostname)"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true

mkdir -p checkpoints outputs

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  --encoder-checkpoint checkpoints/best_simim_25.pt \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_loader_DC_full.json \
  --num-workers 8 \
  --epochs 30 \
  --use-muon \
  --muon-scope matrix \
  --batch-size 4 \
  --accum-steps 2 \
  --compile \
  --channels-last \
  --grad-clip-norm 1.0 \
  --save-optimizer-state \
  --spatial-jitter-px 0 \
  --brightness-jitter 0 \
  --contrast-jitter 0 \
  --drop-path 0.1 \
  --decoder-channels 256 \
  --muon-lr 0.0017414482171862190 \
  --muon-momentum 0.99 \
  --muon-weight-decay 0.017250584504870447 \
  --nadam-lr 0.00029230976233898245 \
  --nadam-beta1 0.9 \
  --nadam-beta2 0.99 \
  --warmup-fraction 0.1 \
  --freeze-encoder-epochs 0 \
  --class-weight-scheme inverse_sqrt \
  --checkpoint-path checkpoints/stage3_full_damped.pt \
  --history-path outputs/stage3_full_damped_history.csv \
  --wandb --wandb-mode online --wandb-project ai4exomars \
  --wandb-group noah-seg-full-damped-lr \
  --wandb-job-type stage3_segmentation \
  --wandb-tags noah segmentation stage3 full-manifest damped-lr grad-clip random-split

echo
echo "Done. Global per-class mIoU (not the per-batch average the loop prints):"
python scripts/eval_global_miou.py checkpoints/stage3_full_damped_30ep.pt
