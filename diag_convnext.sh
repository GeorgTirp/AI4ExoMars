#!/usr/bin/env bash
set -euo pipefail
cd "${AI4EXOMARS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p job_outputs/diag
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true
  module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
[ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "Host: $(hostname)"; nvidia-smi -L || true
PYTHONPATH=".:${PYTHONPATH:-}" python scripts/diag_compile_convnext.py "$@"
