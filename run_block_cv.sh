#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Stage-4 hyperparameter tuning by spatial cross-validation: ONE configuration,
# all folds in sequence (manifest from prep_expert_block_manifest.py --cv-folds).
# Each fold trains on the other train+val folds (minus bordering blocks) and is
# scored at its LAST epoch (--select last: no best-epoch optimism); the test
# split is never read. Writes results/block_cv/<CONFIG>.json (per fold + mean).
#
#   CONFIG=lr1_e12 SEG_LR_SCALE=1 EPOCHS=12 TAU=645 condor_submit_bid 20 run_block_cv.sub
#
# SEG_LR_SCALE multiplies the segmenter LRs of the E13 recipe (Muon 1.72e-5,
# NAdamW 6.08e-6, i.e. 0.1x the tuned Stage-3 values); TAU is the Laplace
# prior strength (0 = no prior); HEAD_LR, LABEL_SMOOTHING optional.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"

: "${CONFIG:?CONFIG is not set}"
MANIFEST="${MANIFEST:-data/BlockLabels_prepared_v2/manifest.csv}"
K="${K:-3}"
EPOCHS="${EPOCHS:-24}"
SEG_LR_SCALE="${SEG_LR_SCALE:-1}"
TAU="${TAU:-0}"
HEAD_LR="${HEAD_LR:-1e-3}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0}"
PRIOR=checkpoints/laplace/noahh_prior_e10.pt
OUT=checkpoints/block_cv/$CONFIG
mkdir -p "$OUT" results/block_cv job_outputs/block_cv

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true; module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MUON_LR=$(python -c "print(1.72e-5 * $SEG_LR_SCALE)")
NADAM_LR=$(python -c "print(6.08e-6 * $SEG_LR_SCALE)")
echo "Host: $(hostname)"
echo "CV config $CONFIG: $K folds, epochs $EPOCHS, segmenter LR x$SEG_LR_SCALE (Muon $MUON_LR, NAdamW $NADAM_LR), tau $TAU, head LR $HEAD_LR, label smoothing $LABEL_SMOOTHING"

for f in $(seq 0 $((K - 1))); do
  python -m vision_backend.train_stage4_block_finetune \
    --manifest "$MANIFEST" --block-classes "$(dirname "$MANIFEST")/block_classes.json" \
    --expect-new-classes 4 --cv-fold "$f" --select last --out-dir "$OUT/fold$f" \
    --epochs "$EPOCHS" --probe-epochs 2 --batch-size 8 --num-workers 8 --seed 42 \
    --muon-lr "$MUON_LR" --nadam-lr "$NADAM_LR" --head-lr "$HEAD_LR" --label-smoothing "$LABEL_SMOOTHING" \
    --laplace-prior "$PRIOR" --laplace-tau "$TAU"
  rm -f "$OUT/fold$f/best.pt"  # CV models are only scored, never reused
done

python - "$OUT" "$K" "$CONFIG" <<'EOF'
import json, sys
import numpy as np
out, k, config = sys.argv[1], int(sys.argv[2]), sys.argv[3]
folds = [json.load(open(f"{out}/fold{f}/metrics.json"))["best_val"] for f in range(k)]
keys = ("accuracy", "balanced_accuracy", "macro_f1", "macro_f1_sel", "nll", "ece")
mean = {key: float(np.nanmean([m[key] for m in folds])) for key in keys}
sd = {key: float(np.nanstd([m[key] for m in folds], ddof=1)) for key in keys}
res = {"config": config, "mean": mean, "sd_over_folds": sd,
       "folds": [{key: m[key] for key in keys} | {"n": m["n"]} for m in folds]}
json.dump(res, open(f"results/block_cv/{config}.json", "w"), indent=2)
print("CV mean: " + "  ".join(f"{k_} {mean[k_]:.4f} (sd {sd[k_]:.4f})" for k_ in keys))
EOF
echo "CV $CONFIG finished cleanly at $(date -Is)."
