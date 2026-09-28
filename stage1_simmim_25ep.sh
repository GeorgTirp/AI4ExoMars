#!/usr/bin/env bash
# Stage-1 SimMIM pretraining — 25 epochs with best sweep hparams.
set -euo pipefail

cd /home/gtirpitz/AI4ExoMars
mkdir -p checkpoints/stage1_simmim_25ep job_outputs/stage1_simmim_25ep

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
nvidia-smi -L || true
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python -m vision_backend.train_stage1_simmim \
  --config vision_backend/configs/stage1_simmim_25ep.yaml \
  --wandb \
  --wandb-project ai4exomars \
  --wandb-group stage1-simmim-25ep \
  --wandb-job-type stage1-simmim-train
