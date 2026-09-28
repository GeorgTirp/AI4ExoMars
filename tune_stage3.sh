#!/usr/bin/env bash
# Append one agent run to an existing Stage-3 wandb sweep.
#
# Usage:
#   ./tune_stage3.sh <entity/ai4exomars/sweep_id>
#   WANDB_SWEEP_ID=<sweep_id> ./tune_stage3.sh
#
# Typical Condor use:
#   condor_submit tune_stage3.sub SWEEP_ID=<entity/ai4exomars/sweep_id>
#
# Initialise the sweep first (once):
#   cd AI4ExoMars
#   wandb sweep --project ai4exomars config/stage3_segmentation_finetune_sweep.yaml
#
# Requires a Stage-2 student checkpoint at:
#   results/tune_stage2/best.pt  (or edit --encoder-checkpoint in the sweep YAML)
set -euo pipefail

SWEEP_ID="${WANDB_SWEEP_ID:-${1:?Pass the sweep id as first arg or set WANDB_SWEEP_ID}}"

cd /home/gtirpitz/AI4ExoMars
mkdir -p results/tune_stage3 job_outputs/tune_stage3

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi

source .venv/bin/activate

prepend_ld_path() {
  if [[ -d "$1" ]]; then
    export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}"
  fi
}
prepend_ld_path "${CUDA_HOME:-/usr/local/cuda}/lib64"
prepend_ld_path "${CUDA_HOME:-/usr/local/cuda}/extras/CUPTI/lib64"

while IFS= read -r lib_dir; do
  prepend_ld_path "${lib_dir}"
done < <(
  python3 - <<'PY'
import pathlib, site, sysconfig
roots = set(site.getsitepackages())
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    roots.add(purelib)
for root in sorted(roots):
    nvidia_root = pathlib.Path(root) / "nvidia"
    if nvidia_root.is_dir():
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            print(lib_dir)
PY
)

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Sweep ID: ${SWEEP_ID}"
nvidia-smi -L || true
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

wandb agent --count "${WANDB_AGENT_COUNT:-1}" "${SWEEP_ID}"
