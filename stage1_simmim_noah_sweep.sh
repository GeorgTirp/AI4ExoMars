#!/usr/bin/env bash
set -euo pipefail

cd /home/gtirpitz/AI4ExoMars

mkdir -p checkpoints/stage1_simmim_noah_sweep job_outputs/stage1_simmim_noah_sweep

# --- fail fast on anything that would make the agent a silent no-op ----------
: "${SWEEP_ID:?SWEEP_ID is not set -- create the sweep first with 'wandb sweep vision_backend/training/sweeps/stage1_simmim_noah.json' and pass it via the .sub 'environment' line}"
: "${WANDB_API_KEY:?WANDB_API_KEY is not set -- 'export WANDB_API_KEY=...' before condor_submit (the .sub uses getenv = True)}"

INDEX_PATH="${INDEX_PATH:-data/noah_simmim_crops/crops_index.csv}"
if [ ! -f "$INDEX_PATH" ]; then
  echo "ERROR: crop index not found: $PWD/$INDEX_PATH" >&2
  echo "Run vision_backend.prep_crops on the NOAH DRG tiles first." >&2
  exit 1
fi

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
  module list || true
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# surface agent-side problems in the job log instead of burying them
export WANDB_SILENT=false
export WANDB_CONSOLE=wrap
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/stage1_simmim_noah_sweep}"

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "Sweep: $SWEEP_ID"
echo "Index: $INDEX_PATH ($(wc -l < "$INDEX_PATH") lines)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi -L || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -c 'import wandb; print("wandb", wandb.__version__)'

# verify the API key actually authenticates before burning a GPU slot
python - <<'PY'
import sys, wandb
try:
    api = wandb.Api()
    print("wandb auth OK as:", api.viewer.entity)
except Exception as exc:
    sys.exit(f"wandb authentication FAILED: {type(exc).__name__}: {exc}")
PY

python -m vision_backend.train_stage1_simmim \
  --config vision_backend/configs/stage1_simmim.yaml \
  --wandb \
  --wandb-mode online \
  --wandb-project "${WANDB_PROJECT:-ai4exomars}" \
  --wandb-group "${WANDB_GROUP:-noah-simmim-tune}" \
  --wandb-job-type stage1_simmim \
  --wandb-tags noah simmim stage1 sweep \
  --wandb-sweep-id "$SWEEP_ID" \
  --wandb-sweep-count "${SWEEP_COUNT:-4}" \
  "$@"

echo "Agent finished cleanly at $(date -Is)."
