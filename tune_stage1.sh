#!/usr/bin/env bash
# Append one agent run to an existing Stage-1 wandb sweep.
#
# Usage:
#   ./tune_stage1.sh <entity/ai4exomars/sweep_id>
#   WANDB_SWEEP_ID=<sweep_id> ./tune_stage1.sh
#
# Typical Condor use:
#   condor_submit tune_stage1.sub SWEEP_ID=<entity/ai4exomars/sweep_id>
#
# Initialise the sweep first (once):
#   cd AI4ExoMars
#   wandb sweep --project ai4exomars config/stage1_teacher_ssl_sweep.yaml
set -euo pipefail

SWEEP_ID="${WANDB_SWEEP_ID:-${1:?Pass the sweep id as first arg or set WANDB_SWEEP_ID}}"

# `wandb agent` on a sweep id that does not resolve fails in seconds without
# ever creating a project or a run, which is indistinguishable at a glance from
# "the job never started". Reject the two ways that actually happened: the
# unedited placeholder from the .sub, and a bare id with no entity/project.
case "$SWEEP_ID" in
  *REPLACE_WITH_SWEEP_ID*|"")
    echo "ERROR: SWEEP_ID is still the placeholder ('$SWEEP_ID')." >&2
    echo "Create the sweep first, then pass its real id:" >&2
    echo "  wandb sweep --project ai4exomars config/stage1_teacher_ssl_sweep.yaml" >&2
    echo "  condor_submit tune_stage1.sub SWEEP_ID=<entity>/<project>/<id>" >&2
    exit 1 ;;
  */*/*) ;;
  *)
    echo "ERROR: SWEEP_ID must be <entity>/<project>/<id>, got '$SWEEP_ID'." >&2
    echo "A bare id resolves against the default entity and usually 404s." >&2
    exit 1 ;;
esac

# Same contract as run_variant_sweep.sh: WANDB_API_KEY or a ~/.netrc entry.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'api\.wandb\.ai' "${HOME}/.netrc"; then
  echo "ERROR: no wandb credentials -- run 'wandb login' or export WANDB_API_KEY." >&2
  exit 1
fi

cd /home/gtirpitz/AI4ExoMars
mkdir -p results/tune_stage1 job_outputs/tune_stage1

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

# Trials per job. tune_stage1.sub passes this as the second argument
# (`arguments = $(SWEEP_ID) 4`); before this it was dropped on the floor and
# every agent silently ran a single trial instead of the four it was asked for.
# WANDB_AGENT_COUNT still wins when set explicitly.
AGENT_COUNT="${WANDB_AGENT_COUNT:-${2:-1}}"
echo "Trials this agent will run: ${AGENT_COUNT}"
wandb agent --count "${AGENT_COUNT}" "${SWEEP_ID}"
