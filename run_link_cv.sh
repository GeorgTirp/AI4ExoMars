#!/usr/bin/env bash
set -euo pipefail
# E18: output link of the Stage-4 head (softmax + CE vs normcdf + per-class BCE), by 3-fold spatial CV
# at the final recipe (12 epochs, segmenter LR x3, tau 64.5, last epoch). Per fold: train, evaluate the
# uncertainty by cross-fitting (scripts/eval_link_cv_fold.py), delete the fold model.
#   LINK=normcdf condor_submit_bid 20 run_link_cv.sub
# Afterwards, on the login node: python scripts/summarize_link_cv.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
: "${LINK:?LINK is not set (softmax | normcdf)}"
MANIFEST="${MANIFEST:-data/BlockLabels_prepared_v2/manifest.csv}"
K="${K:-3}"
OUT=checkpoints/block_link_cv/$LINK
mkdir -p "$OUT" results/block_link_cv
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true; module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Host: $(hostname)  link $LINK  $K folds"
for f in $(seq 0 $((K - 1))); do
  python -m vision_backend.train_stage4_block_finetune \
    --manifest "$MANIFEST" --block-classes "$(dirname "$MANIFEST")/block_classes.json" \
    --expect-new-classes 4 --cv-fold "$f" --select last --out-dir "$OUT/fold$f" --link "$LINK" \
    --epochs 12 --probe-epochs 2 --batch-size 8 --num-workers 8 --seed 42 \
    --muon-lr 5.16e-5 --nadam-lr 1.824e-5 --head-lr 1e-3 \
    --laplace-prior checkpoints/laplace/noahh_prior_e10.pt --laplace-tau 64.5
  python scripts/eval_link_cv_fold.py "$OUT/fold$f/best.pt" --fold "$f" --out "results/block_link_cv/$LINK.fold$f"
  rm -f "$OUT/fold$f/best.pt"  # CV models are only scored, never reused
done
echo "Link CV $LINK finished cleanly at $(date -Is)."
