#!/usr/bin/env bash
set -euo pipefail

cd /home/gtirpitz/AI4ExoMars

mkdir -p results/tune_stage1_50ep job_outputs/stage1_training_50ep

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/13.0 || module load cuda/12.6 || module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
  module list || true
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

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi -L || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python -m vision_backend.train_stage1_teacher_ssl \
  --index-path data/patch_index.csv \
  --dataset-backend auto \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 4 \
  --val-fraction 0.1 \
  --test-fraction 0 \
  --local-input-size 256 \
  --context-input-size 256 \
  --local-base-channels 48 \
  --context-base-channels 24 \
  --context-dim 256 \
  --decoder-channels 256 \
  --swin-depths 2 2 2 \
  --swin-num-heads 4 8 16 \
  --window-size 8 \
  --drop-path 0 \
  --mask-patch-size 8 \
  --mask-ratio 0.5 \
  --loss-type l1 \
  --use-muon \
  --learning-rate 0.0018792598465717752 \
  --weight-decay 0.0020406330520434737 \
  --warmup-fraction 0.11011581589732208 \
  --adam-beta1 0.9438701951139566 \
  --adam-beta2 0.950022063641192 \
  --adam-eps 1.9904139414832822e-9 \
  --muon-momentum 0.8975859368554203 \
  --muon-ns-steps 5 \
  --seed 42 \
  --initial-checkpoint-path results/tune_stage1_50ep/initial.pt \
  --checkpoint-path results/tune_stage1_50ep/best.pt \
  --final-checkpoint-path results/tune_stage1_50ep/final.pt \
  --history-path results/tune_stage1_50ep/history.csv \
  --examples-path results/tune_stage1_50ep/examples.pt \
  --num-examples 5 \
  "$@"
