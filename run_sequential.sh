#!/usr/bin/env bash
set -euo pipefail
# E19: two labelling campaigns in sequence with the diagonal Laplace filter (prior -> posterior -> prior;
# Sliwa et al., arXiv:2410.06800, diagonal version), simulated with the two expert observations:
# campaign A = ESP_075512 (512 blocks), campaign B = ESP_027201 (1024 blocks). Final Stage-4 recipe.
#   1. A from the Stage-3 segmenter with the NOAH-H prior; its posterior (tau F_NOAH-H + N_A F_A).
#   2. B from model A: "naive" (no prior) and "filtered" (A's posterior as the prior, tau 1).
# Every model is scored on both observations' test blocks and on NOAH-H; the joint model
# (v2_final_lr3_e12, A and B together) is the reference.
#   condor_submit_bid 20 run_sequential.sub
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true; module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Host: $(hostname)"
V2=data/BlockLabels_prepared_v2
A=checkpoints/block_finetune/seqA_075512
POST=checkpoints/laplace/seqA_075512_posterior.pt
RECIPE=(--expect-new-classes 4 --select last --epochs 12 --probe-epochs 2 --batch-size 8 --num-workers 8
        --seed 42 --muon-lr 5.16e-5 --nadam-lr 1.824e-5 --head-lr 1e-3)
mkdir -p results/block_sequential

python -m vision_backend.train_stage4_block_finetune --manifest data/BlockLabels_prepared/manifest.csv \
  --block-classes data/BlockLabels_prepared/block_classes.json --out-dir "$A" "${RECIPE[@]}" \
  --laplace-prior checkpoints/laplace/noahh_prior_e10.pt --laplace-tau 64.5
python scripts/eval_block_classifier.py "$A/best.pt" --manifest "$V2/manifest.csv" --out "$A/eval_v2.json" --noahh-eval
python scripts/fit_campaign_posterior.py "$A/best.pt" --out "$POST"

for ARM in naive filtered; do
  OUT=checkpoints/block_finetune/seqB_027201_$ARM
  PRIOR=()
  if [ "$ARM" = filtered ]; then PRIOR=(--laplace-prior "$POST" --laplace-tau 1); fi
  python -m vision_backend.train_stage4_block_finetune --manifest "$V2/manifest_ESP_027201.csv" \
    --block-classes "$V2/block_classes.json" --out-dir "$OUT" --init-block-checkpoint "$A/best.pt" \
    "${RECIPE[@]}" ${PRIOR[@]+"${PRIOR[@]}"}
  python scripts/eval_block_classifier.py "$OUT/best.pt" --manifest "$V2/manifest.csv" --out "$OUT/eval_v2.json" --noahh-eval
done
for R in seqA_075512 seqB_027201_naive seqB_027201_filtered; do
  cp "checkpoints/block_finetune/$R/eval_v2.json" "results/block_sequential/$R.eval_v2.json"
done
echo "Sequential campaigns finished cleanly at $(date -Is)."
