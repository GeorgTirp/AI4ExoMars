#!/usr/bin/env bash
set -euo pipefail
# Re-evaluate an existing block fine-tune (scripts/eval_block_classifier.py, with NOAH-H forgetting).
#   RUN_NAME=experts_v1_plain condor_submit_bid 20 run_block_eval.sub
#   RUN_NAME=laplace_all_t65 MANIFEST=data/BlockLabels_prepared_v2/manifest.csv EVAL_TAG=v2 NOAHH=0 \
#     condor_submit_bid 20 run_block_eval.sub     (another test set; NOAHH=0 skips the NOAH-H eval)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
: "${RUN_NAME:?RUN_NAME is not set}"
MANIFEST="${MANIFEST:-data/BlockLabels_prepared/manifest.csv}"
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true; module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "Host: $(hostname)"
# EVAL_TAG names the output (eval_<tag>.json), so evaluating a run on another
# manifest does not overwrite its own evaluation.
NAME="eval${EVAL_TAG:+_$EVAL_TAG}"
NOAHH_ARGS=(--noahh-eval)
[ "${NOAHH:-1}" = "0" ] && NOAHH_ARGS=()
python scripts/eval_block_classifier.py "checkpoints/block_finetune/$RUN_NAME/best.pt" --manifest "$MANIFEST" \
  --out "checkpoints/block_finetune/$RUN_NAME/$NAME.json" ${NOAHH_ARGS[@]+"${NOAHH_ARGS[@]}"}
cp "checkpoints/block_finetune/$RUN_NAME/$NAME.json" "results/block_finetune/${RUN_NAME}.$NAME.json"
echo "Eval $RUN_NAME finished cleanly at $(date -Is)."
