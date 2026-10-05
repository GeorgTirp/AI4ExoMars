#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Architecture ablations of the HybridEncoder at the tuned recipe.
#
# Reference: tuning trial f1qeno3u (sweep jjo7a5f6), val mIoU 0.2060 -- same
# code, seed 42, data order, 16 epochs, Muon (transformer matrices, match_adam)
# + NAdamW, Lovász, drop-path, and exactly its hyperparameters (below).
#
#   ABLATION=no_global  S4 global-attention block -> shifted-window Swin block
#                       (27.57 M vs 27.66 M encoder params: param-matched)
#   ABLATION=s3_depth2  6 -> 2 Swin blocks at 1/16 (20.55 M encoder params,
#                       -26 %: depth and capacity drop together)
#   ABLATION=none       the reference itself (a re-run / seed control)
#   ABLATION=no_global_s3_depth2  both architecture ablations together
#   ABLATION=no_lovasz  recipe ablation: Lovász weight 0 (class-weighted CE only)
#   ABLATION=no_droppath  recipe ablation: drop-path 0
#
#   ABLATION=no_global condor_submit_bid 20 run_arch_ablation.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

: "${ABLATION:?ABLATION is not set -- no_global, s3_depth2, no_global_s3_depth2, no_lovasz, no_droppath or none}"
LOVASZ_WEIGHT=0.49114339221744663
DROP_PATH=0.08716622036277302
case "$ABLATION" in
  no_global)           ARCH_ARGS=(--hybrid-s4-block swin) ;;
  s3_depth2)           ARCH_ARGS=(--hybrid-s3-depth 2) ;;
  no_global_s3_depth2) ARCH_ARGS=(--hybrid-s4-block swin --hybrid-s3-depth 2) ;;
  no_lovasz)           ARCH_ARGS=(); LOVASZ_WEIGHT=0.0 ;;
  no_droppath)         ARCH_ARGS=(); DROP_PATH=0.0 ;;
  none)                ARCH_ARGS=() ;;
  *) echo "ERROR: unknown ABLATION '$ABLATION'" >&2; exit 1 ;;
esac
SEED="${SEED:-42}"
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'api\.wandb\.ai' "${HOME}/.netrc"; then
  echo "ERROR: no wandb credentials found (wandb login, or export WANDB_API_KEY)." >&2
  exit 1
fi

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="$DER/seg_loader_DC_full.json"
CROP_CACHE_DIR="$DER/seg_crop_cache_full"
[ -f "$CROP_CACHE_DIR/meta.json" ] || { echo "ERROR: crop cache missing: $CROP_CACHE_DIR" >&2; exit 1; }

OUT_TAG="arch_ablation_${ABLATION}"
mkdir -p "checkpoints/${OUT_TAG}" job_outputs/arch_ablation results/arch_ablation

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
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/arch_ablation}"
export WANDB_CONSOLE=wrap

python -c "import muon" || { echo "ERROR: muon not installed in the venv" >&2; exit 1; }
echo "Host: $(hostname)"
echo "Ablation: $ABLATION  (${ARCH_ARGS[*]:-frozen layout})  drop-path $DROP_PATH  lovasz $LOVASZ_WEIGHT  seed $SEED"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  --random-init-encoder \
  --global-base-grid 32 \
  --window-size 8 \
  ${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"} \
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
  --lovasz-weight "$LOVASZ_WEIGHT" \
  --ig-loss-weight 0.4 \
  --ema-decay 0.9999 \
  --seed "$SEED" \
  --compile \
  --amp-dtype bf16 \
  --variant-id "arch_${ABLATION}_s${SEED}" \
  --variant-metrics-path "results/arch_ablation/${ABLATION}_s${SEED}.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --wandb \
  --wandb-mode online \
  --wandb-project ai4exomars_arch_ablation \
  --wandb-group arch-ablation \
  --wandb-job-type "$ABLATION" \
  --wandb-tags arch-ablation "$ABLATION" simmim muon lovasz

echo "Ablation $ABLATION (seed $SEED) finished cleanly at $(date -Is)."
