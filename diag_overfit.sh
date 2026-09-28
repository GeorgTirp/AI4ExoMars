#!/usr/bin/env bash
set -euo pipefail

# Runs scripts/diag_overfit_one_batch.py on a GPU slot -- see that file's
# docstring for what the three arms separate.
#
#   condor_submit_bid 20 diag_overfit.sub

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

mkdir -p job_outputs/diag results/diag

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

echo "Host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

# PYTHONPATH=. because scripts/ has no __init__.py, so `python -m scripts....`
# would not resolve and `python scripts/x.py` puts scripts/ (not the repo root)
# on sys.path, breaking `import vision_backend`.
PYTHONPATH="${PYTHONPATH:-}:." python scripts/diag_overfit_one_batch.py \
  --batch "${BATCH:-8}" \
  --steps "${STEPS:-300}" \
  --lr "${LR:-1e-3}" \
  "$@"

echo "Diagnostic finished at $(date -Is)."
