#!/usr/bin/env bash
set -euo pipefail

cd /home/gtirpitz/AI4ExoMars

mkdir -p results/stage1_training job_outputs/stage1_training

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

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi -L || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python -m vision_backend.train_stage1_teacher_ssl \
  --index-path hirise_context_pairs/patch_index.csv \
  --dataset-backend auto \
  --epochs 10 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  --test-fraction "${TEST_FRACTION:-0.1}" \
  --use-muon \
  --initial-checkpoint-path results/stage1_training/stage1_initial_model.pt \
  --checkpoint-path results/stage1_training/stage1_best_model.pt \
  --final-checkpoint-path results/stage1_training/stage1_final_model.pt \
  --history-path results/stage1_training/stage1_loss_trajectory.csv \
  --examples-path results/stage1_training/stage1_examples.pt \
  "$@"
