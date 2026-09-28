#!/usr/bin/env bash
set -euo pipefail
# SERVER_LAUNCH.md step 2b follow-up: the measured step ratio was 0.758, outside
# the 0.68-0.72 band. decoder_channels moves the step, local_base_channels moves
# params -- grid both and report where each candidate lands on BOTH ratios.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AI4EXOMARS_ROOT:-$SCRIPT_DIR}"
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh; module purge || true
  module load cuda/12.1 || true; module load cudnn/9.10.2 || true
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "Host: $(hostname)"; nvidia-smi -L || true
python - <<'EOF'
import statistics, time, torch
from vision_backend.training.builders import build_context_segmentation_model
dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
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
print(f"\nBASELINE big: {pb:,} params  {tb:.1f} ms\n")
print(f"{'lbc':>4} {'cbc':>4} {'cdim':>5} {'dch':>4} {'params':>12} {'ms':>7} {'p_ratio':>8} {'t_ratio':>8}  band")
rows=[]
for lbc in (44, 46, 48, 50):
    cbc = round(lbc/2); cdim = round(256*lbc/52)
    for dch in (144, 160, 176, 192):
        ps, ts = bench(cfg(lbc, cbc, cdim, dch))
        pr, tr = ps/pb, ts/tb
        ok = "OK" if (0.68<=pr<=0.72 and 0.68<=tr<=0.72) else ""
        print(f"{lbc:>4} {cbc:>4} {cdim:>5} {dch:>4} {ps:>12,} {ts:>7.1f} {pr:>8.3f} {tr:>8.3f}  {ok}", flush=True)
        rows.append((abs(pr-0.70)+abs(tr-0.70), lbc, cbc, cdim, dch, ps, ts, pr, tr))
rows.sort()
print("\nclosest to 0.70 on both:")
for r in rows[:4]:
    print(f"  lbc={r[1]} cbc={r[2]} cdim={r[3]} dch={r[4]}  params={r[5]:,} ({r[7]:.3f})  step={r[6]:.1f}ms ({r[8]:.3f})")
EOF
echo "Calibration finished at $(date -Is)."
