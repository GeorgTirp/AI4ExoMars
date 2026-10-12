#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Stage 4: block-level fine-tuning (one class per 512x512 block) of the tuned
# HybridEncoder segmenter -- vision_backend/train_stage4_block_finetune.py.
#
#   Expert labels (scripts/prep_expert_block_manifest.py -> data/BlockLabels_prepared):
#     RUN_NAME=experts_v1 MANIFEST=data/BlockLabels_prepared/manifest.csv \
#       condor_submit_bid 20 run_block_finetune.sub
#   Dry run on NOAH-H stand-in labels (dominant DC class per block, DC 3, 7,
#   10, 11 relabelled as "new" classes) -- built inside the job:
#     RUN_NAME=dryrun DRYRUN=1 condor_submit_bid 20 run_block_finetune.sub
#
# Optional: SEG_CKPT (default: E10 full hybrid, val mIoU 0.2072), EXPECT_NEW
# (default 4: the four new expert classes), EPOCHS, PROBE_EPOCHS, EXTRA_ARGS.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

: "${RUN_NAME:?RUN_NAME is not set}"
SEG_CKPT="${SEG_CKPT:-checkpoints/arch_ablation_no_lovasz/best_16ep.pt}"
EXPECT_NEW="${EXPECT_NEW:-4}"
EPOCHS="${EPOCHS:-30}"
PROBE_EPOCHS="${PROBE_EPOCHS:-2}"
OUT_DIR="checkpoints/block_finetune/${RUN_NAME}"
mkdir -p "$OUT_DIR" job_outputs/block_finetune results/block_finetune

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
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/block_finetune}"
export WANDB_CONSOLE=wrap

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ] || grep -qs 'api\.wandb\.ai' "${HOME}/.netrc"; then
  WANDB_ARGS=(--wandb --wandb-mode online --wandb-project ai4exomars_block_finetune
              --wandb-name "$RUN_NAME" --wandb-tags stage4 block "$RUN_NAME")
fi

if [ "${DRYRUN:-0}" = "1" ]; then
  MANIFEST="$OUT_DIR/noahh_blocks.csv"
  python scripts/make_noahh_block_labels.py --out "$MANIFEST" --per-split 4000 1000 --as-new 3 7 10 11
fi
: "${MANIFEST:?MANIFEST is not set (or use DRYRUN=1)}"
# Fixed class list from scripts/prep_expert_block_manifest.py, when present.
CLASS_ARGS=()
BLOCK_CLASSES="$(dirname "$MANIFEST")/block_classes.json"
[ -f "$BLOCK_CLASSES" ] && CLASS_ARGS=(--block-classes "$BLOCK_CLASSES")
[ -f "$SEG_CKPT" ] || { echo "ERROR: segmenter checkpoint missing: $SEG_CKPT" >&2; exit 1; }
python -c "import muon" || { echo "ERROR: muon not installed in the venv" >&2; exit 1; }

echo "Host: $(hostname)"
echo "Block fine-tune: $RUN_NAME  manifest $MANIFEST  segmenter $SEG_CKPT  epochs $EPOCHS (probe $PROBE_EPOCHS)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

# EXTRA_ARGS is split on spaces on purpose (extra CLI flags).
# shellcheck disable=SC2086
python -m vision_backend.train_stage4_block_finetune \
  --seg-checkpoint "$SEG_CKPT" \
  --manifest "$MANIFEST" \
  ${CLASS_ARGS[@]+"${CLASS_ARGS[@]}"} \
  --expect-new-classes "$EXPECT_NEW" \
  --out-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --probe-epochs "$PROBE_EPOCHS" \
  --batch-size 8 \
  --num-workers 8 \
  --seed 42 \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  ${EXTRA_ARGS:-}

cp "$OUT_DIR/metrics.json" "results/block_finetune/${RUN_NAME}.json"
echo "Block fine-tune $RUN_NAME finished cleanly at $(date -Is)."
