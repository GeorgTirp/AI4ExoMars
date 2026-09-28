#!/usr/bin/env bash
set -euo pipefail

cd /home/gtirpitz/AI4ExoMars

mkdir -p results/stage1_training job_outputs/stage1_training

# Load cuda/12.1 first to get system libnccl.so.2 (torch._C is hard-linked
# against it and fails to import without it, even on single-GPU).
# Load cudnn/9.10.2 afterwards to override the older cudnn that cuda/12.1
# auto-loads, mirroring the approach used in the JAX training script.
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi

source .venv/bin/activate

# Add bundled CUDA runtime libs from PyTorch's nvidia-*-cu12 wheel packages.
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
export CUDA_LAUNCH_BLOCKING=1  # serialise kernel launches so the real error location shows in the traceback

# If Condor didn't pin the GPU, default to 0 so we don't accidentally share with other jobs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

# batch_size=128 targets ~22-25 GB on a 40 GB A100 (dual-branch Swin, 256x256, AMP).
# Raise to 192 or 256 if nvidia-smi shows substantial headroom after the first few steps.
python3 -m vision_backend.train_stage1_teacher_ssl \
  --index-path data/patch_index.csv \
  --dataset-backend auto \
  --epochs 50 \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  --test-fraction 0.0 \
  --log-every-steps 5000 \
  --use-muon \
  --initial-checkpoint-path results/stage1_training/stage1_1ep_initial_model.pt \
  --checkpoint-path results/stage1_training/stage1_1ep_best_model.pt \
  --final-checkpoint-path results/stage1_training/stage1_1ep_final_model.pt \
  --history-path results/stage1_training/stage1_1ep_loss_trajectory.csv \
  --examples-path results/stage1_training/stage1_1ep_examples.pt \
  "$@"
