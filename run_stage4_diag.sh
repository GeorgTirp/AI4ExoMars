#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Stage-4 diagnostics + Laplace prior fit (one GPU job, ~20 min):
#   1. E13 (plain block fine-tune): temperature scaling, spatial-bootstrap CIs,
#      val-vs-test on the same classes, NOAH-H val mIoU of its segmenter
#   2. the E10 segmenter's NOAH-H val mIoU with the same evaluator (reference)
#   3. zero-shot segmenter on the expert val blocks low-passed to the Oxia
#      spectrum (sigma 0.7 / 1.0 px; scripts/diag_site_shift.py)
#   4. campaign-0 Laplace prior: per-image MC Fisher on 2000 NOAH-H crops, and
#      the tau grid calibrated on E13's weight change
#
#   condor_submit_bid 20 run_stage4_diag.sub
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load cuda/12.1 || true
  module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DER=data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived
E13=checkpoints/block_finetune/experts_v1_plain/best.pt
echo "Host: $(hostname)"; nvidia-smi -L || true
# The steps are independent: run them all, fail the job at the end if any failed.
FAILED=()
step() { local name="$1"; shift; echo "=== $name"; "$@" || { echo "!!! step failed: $name"; FAILED+=("$name"); }; }

step "1. E13 evaluation" bash -c "python scripts/eval_block_classifier.py $E13 \
  --manifest data/BlockLabels_prepared/manifest.csv --noahh-eval && \
  cp checkpoints/block_finetune/experts_v1_plain/eval.json results/block_finetune/experts_v1_plain.eval.json"

step "2. E10 segmenter, same NOAH-H evaluator" python scripts/eval_global_miou.py checkpoints/arch_ablation_no_lovasz/best_16ep.pt --batch-size 8 \
  --loader-config-path "$DER/seg_loader_DC_full.json" --crop-cache-dir "$DER/seg_crop_cache_full" \
  --json-out results/block_finetune/e10_noahh_val_miou.json

for s in 0.7 1.0; do
  step "3. zero-shot, blur $s" python -m vision_backend.train_stage4_block_finetune --eval-only --num-workers 4 \
    --manifest "data/BlockLabels_prepared_blur$s/manifest.csv" \
    --block-classes "data/BlockLabels_prepared_blur$s/block_classes.json" \
    --expect-new-classes 4 --out-dir "checkpoints/block_finetune/zeroshot_blur$s"
  cp "checkpoints/block_finetune/zeroshot_blur$s/metrics.json" "results/block_finetune/zeroshot_blur$s.json" || true
done

step "4. Laplace prior" python scripts/fit_laplace_prior.py --n-images 2000 --out checkpoints/laplace/noahh_prior_e10.pt \
  --calibrate-with "$E13"
cp checkpoints/laplace/noahh_prior_e10.calibration.json results/block_finetune/laplace_prior_calibration.json || true

if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED steps: ${FAILED[*]}" >&2; exit 1; fi
echo "Stage-4 diagnostics finished cleanly at $(date -Is)."
