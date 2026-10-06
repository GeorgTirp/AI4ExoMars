#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Sweep agent: tune the production candidate (config/tune_simmim_muon_sweep.yaml)
#   SimMIM HybridEncoder, random init
#   + Muon on the 32 transformer matrices (match_adam LR transfer), NAdamW rest
#   + stochastic depth
#   16 epochs per trial (the 30-epoch Muon run peaked at epoch 16).
#
# The sweep sets muon_lr, nadam_lr, muon_weight_decay, nadam_weight_decay
# and drop_path per trial (Lovász was swept in the pilot, then retired); the
# values below are only the base config they override. Everything else matches the E4/E5 runs.
#
#   wandb sweep --project ai4exomars_tune_simmim config/tune_simmim_muon_sweep.yaml
#   SWEEP_ID=<entity/project/id> NTRIALS=4 condor_submit_bid 20 run_tune_simmim.sub
# Each queued job is one agent running SWEEP_COUNT (default 1) trials.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

: "${SWEEP_ID:?SWEEP_ID is not set -- create it with 'wandb sweep --project ai4exomars_tune_simmim config/tune_simmim_muon_sweep.yaml'}"
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'api\.wandb\.ai' "${HOME}/.netrc"; then
  echo "ERROR: no wandb credentials found (wandb login, or export WANDB_API_KEY)." >&2
  exit 1
fi

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
LOADER_CONFIG="$DER/seg_loader_DC_full.json"
CROP_CACHE_DIR="$DER/seg_crop_cache_full"
[ -f "$CROP_CACHE_DIR/meta.json" ] || { echo "ERROR: crop cache missing: $CROP_CACHE_DIR" >&2; exit 1; }

OUT_TAG="tune_simmim"
mkdir -p "checkpoints/${OUT_TAG}" "job_outputs/${OUT_TAG}" "results/${OUT_TAG}"

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
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/${OUT_TAG}}"
export WANDB_CONSOLE=wrap

python -c "import muon" || { echo "ERROR: muon not installed in the venv" >&2; exit 1; }
echo "Host: $(hostname)"
echo "Sweep: $SWEEP_ID  (trials this agent: ${SWEEP_COUNT:-1})"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  --random-init-encoder \
  --global-base-grid 32 \
  --window-size 8 \
  --drop-path 0.1 \
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
  --muon-lr 1.107e-4 \
  --muon-weight-decay 5.16e-5 \
  --muon-momentum 0.95 \
  --nadam-lr 1.107e-4 \
  --nadam-weight-decay 5.16e-5 \
  --ig-loss-weight 0.4 \
  --ema-decay 0.9999 \
  --seed 42 \
  --compile \
  --amp-dtype bf16 \
  --per-run-checkpoint \
  --variant-id tune_simmim \
  --variant-metrics-path "results/${OUT_TAG}/trials.jsonl" \
  --checkpoint-path "checkpoints/${OUT_TAG}/best.pt" \
  --wandb \
  --wandb-mode online \
  --wandb-project ai4exomars_tune_simmim \
  --wandb-group tune-simmim-muon \
  --wandb-job-type tune \
  --wandb-tags tune simmim muon droppath \
  --wandb-sweep-id "$SWEEP_ID" \
  --wandb-sweep-count "${SWEEP_COUNT:-1}"

echo "Agent finished cleanly at $(date -Is)."
