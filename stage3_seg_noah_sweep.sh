#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Stage-3 segmentation sweep agent (single-branch SimMIM encoder + NOAH-H data).
#
# ONE-TIME pre-submit steps (run on the server, NOT per agent):
#   1. Transfer derived/drg_on_label_grid.tif + labels_DC_classid.tif + the
#      encoder checkpoint into place.
#   2. Build the crop manifest (fast; do it once, agents must not race on it):
#        python -m vision_backend.prep_seg_crops \
#          --imagery data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/drg_on_label_grid.tif \
#          --labels  data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/labels_DC_classid.tif \
#          --crop-size 512 --min-label-frac 0.5 --val-fraction 0.15 \
#          --out data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_crops_DC_full.csv
#   3. Write the loader config JSON (server paths) at $LOADER_CONFIG (see below).
#   4. Create the sweep and export its id + your key, then submit:
#        export WANDB_API_KEY=...
#        export SWEEP_ID=$(wandb sweep vision_backend/training/sweeps/stage3_segmentation_finetune.json 2>&1 | awk '/Creating sweep with ID/{print $NF}')
#        condor_submit stage3_seg_noah_sweep.sub
# ---------------------------------------------------------------------------

cd /home/gtirpitz/AI4ExoMars

mkdir -p checkpoints/stage3_seg_noah_sweep job_outputs/stage3_seg_noah_sweep

# --- fail fast on anything that would make the agent a silent no-op ----------
: "${SWEEP_ID:?SWEEP_ID is not set -- create it with 'wandb sweep vision_backend/training/sweeps/stage3_segmentation_finetune.json' and export it before condor_submit}"
: "${WANDB_API_KEY:?WANDB_API_KEY is not set -- 'export WANDB_API_KEY=...' before condor_submit (the .sub uses getenv = True)}"

DER="${DER:-data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived}"
MANIFEST="${MANIFEST:-$DER/seg_crops_DC_full.csv}"
LOADER_CONFIG="${LOADER_CONFIG:-$DER/seg_loader_DC_full.json}"
ENCODER_CKPT="${ENCODER_CKPT:-best_simim_25.pt}"

for f in "$DER/drg_on_label_grid.tif" "$DER/labels_DC_classid.tif" \
         "$MANIFEST" "$LOADER_CONFIG" "$ENCODER_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required input not found: $PWD/$f" >&2
    echo "Complete the one-time pre-submit steps at the top of this script." >&2
    exit 1
  fi
done

# --- CUDA / cuDNN (fixes 'libcudnn.so.9: cannot open shared object file') -----
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
export WANDB_SILENT=false
export WANDB_CONSOLE=wrap
export WANDB_DIR="${WANDB_DIR:-$PWD/job_outputs/stage3_seg_noah_sweep}"

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "Sweep: $SWEEP_ID"
echo "Manifest: $MANIFEST ($(wc -l < "$MANIFEST") lines)"
echo "Loader config: $LOADER_CONFIG"
echo "Encoder checkpoint: $ENCODER_CKPT"
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

# NOTE: run as a module from the repo root, so the loader factory must be
# addressed as 'vision_backend.seg_dataset:...' (the bare 'seg_dataset:...'
# default only resolves when cwd is inside vision_backend/).
python -m vision_backend.train_stage3_segmentation_finetune \
  --model-kind simmim \
  --encoder-checkpoint "$ENCODER_CKPT" \
  --loader-factory vision_backend.seg_dataset:create_segmentation_dataloaders \
  --loader-config-path "$LOADER_CONFIG" \
  --num-workers "${NUM_WORKERS:-8}" \
  --epochs "${EPOCHS:-50}" \
  --wandb \
  --wandb-mode online \
  --wandb-project "${WANDB_PROJECT:-ai4exomars}" \
  --wandb-group "${WANDB_GROUP:-noah-seg-tune}" \
  --wandb-job-type stage3_segmentation \
  --wandb-tags noah segmentation stage3 sweep \
  --wandb-sweep-id "$SWEEP_ID" \
  --wandb-sweep-count "${SWEEP_COUNT:-4}" \
  "$@"

echo "Agent finished cleanly at $(date -Is)."
