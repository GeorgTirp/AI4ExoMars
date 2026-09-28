#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# SERVER_LAUNCH.md step 2b -- measure the small/big step-time ratio ON THIS
# HARDWARE before spending four sweeps' worth of GPU time on the 0.68-0.72x
# claim. Needs a GPU, so it runs as a Condor job, not on the login node.
#
#   condor_submit_bid <bid> bench_variant_steptime.sub
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

echo "Host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python - <<'EOF'
import statistics, time, torch
from vision_backend.training.builders import build_context_segmentation_model
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
dev = torch.device("cuda")
def cfg(lbc, cbc, cdim, dch):
    return dict(in_channels=1, local_base_channels=lbc, context_base_channels=cbc,
                context_dim=cdim, use_stage32=True, swin_depths=(2,2,2),
                swin_num_heads=(4,8,16), window_size=8, drop_path=0.0, num_classes=14,
                decoder_channels=dch, use_context=False)
def bench(c, bs=4, n=8):
    m = build_context_segmentation_model(c).to(dev)
    x = torch.randn(bs,1,512,512, device=dev); o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    def step():
        o.zero_grad(); y = m(x)
        (y[0] if isinstance(y, tuple) else y).float().mean().backward(); o.step()
    for _ in range(3): step()
    torch.cuda.synchronize(); ts = []
    for _ in range(5):
        t = time.time()
        for _ in range(n): step()
        torch.cuda.synchronize(); ts.append((time.time()-t)/n*1000)
    p = sum(q.numel() for q in m.parameters())
    del m, o, x; torch.cuda.empty_cache()
    return p, statistics.median(ts)
pb, tb = bench(cfg(52,26,256,256))
ps, ts = bench(cfg(44,22,217,192))
print(f"big   {pb:,} params  {tb:.0f} ms")
print(f"small {ps:,} params  {ts:.0f} ms")
print(f"ratios: params {ps/pb:.3f}  step {ts/tb:.3f}")
EOF

echo "Benchmark finished at $(date -Is)."
