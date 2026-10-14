#!/usr/bin/env bash
set -euo pipefail
# Uncertainty / explainability files of a Stage-4 block classifier (scripts/fit_block_artifacts.py):
# last-layer Laplace, Mahalanobis and neural-PCA bases (block and pixel level), the segmenter in
# Stage-3 format and the campaign posterior. Needs a GPU, the expert blocks and the NOAH-H crops.
#   RUN_NAME=v2_final_lr3_e12 condor_submit_bid 20 run_block_artifacts.sub
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
: "${RUN_NAME:?RUN_NAME is not set}"
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true; module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
python scripts/fit_block_artifacts.py "checkpoints/block_finetune/$RUN_NAME/best.pt" ${EXTRA_ARGS:-}
echo "Block artifacts $RUN_NAME finished cleanly at $(date -Is)."
