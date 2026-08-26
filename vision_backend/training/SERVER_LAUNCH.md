# Server launch runbook — model-variant sweeps (run TOMORROW, on the server)

Runbook for the **server** coding agent. All code is already implemented and pulled; nothing here
was started on the local machine. Your job: build the context cache, initialize the four wandb
sweeps, submit the runs, then collect the comparison.

> **Finalized by the local implementation session** (MODEL_VARIANTS_EXPERIMENT.md §7): every
> command below is the real one as implemented, verified offline. Nothing was launched locally.
>
> **One deviation from the brief you should know about.** §C2 asked for `small` to be 0.68-0.72x
> `big` in *both* params and step time while §5 held `decoder_channels=256` fixed. Those cannot
> both hold: the fixed-width decoder is ~73% of a training step (measured: encoder 209 ms vs
> decoder 561 ms of the forward at 512^2), so encoder width alone moved wall-clock by 3%.
> Per the user's decision, `decoder_channels` is scaled too (256 -> 192 for the small variants).
> **Consequence: V0-vs-V2 compares whole-model width, not encoder width alone** -- say so in the
> writeup.

## 0. Prereqs (verify before anything)
- Repo pulled to the intended commit; Python env active (`uv`/venv as usual).
- Data present on the server:
  - warped imagery on the label grid:
    `data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/drg_on_label_grid.tif`
  - seg manifest, label raster, loader config, local crop cache (all under that same
    `derived/` dir, referred to below as `$DER`):
    `seg_crops_DC_full.csv`, `labels_DC_classid.tif`, `seg_loader_DC_full.json`,
    `seg_crop_cache_full/images.npy`
- GPU + scheduler available (Condor `condor_submit_bid`, or plain `wandb agent`).
- Quick sanity: `pytest -q` (or the variant smoke tests) is green.

## 1. Build the context crop cache — SINGLE CALL (needed only for V1 + V3)
Run once; produces the context tensors index-aligned to `images.npy`. **V0/V2 do not need this.**

```bash
cd ~/AI4ExoMars
DER=data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived

python -m vision_backend.prep_seg_context_cache \
  --manifest            $DER/seg_crops_DC_full.csv \
  --imagery             $DER/drg_on_label_grid.tif \
  --context-size        2048 \
  --context-output-size 512 \
  --out-dir             $DER/seg_context_cache_full
```

Writes `$DER/seg_context_cache_full/context.npy` (uint8, shape `(N, 512, 512)`, index-aligned to
`images.npy`) plus `context_meta.json`.

**Size/time:** one row is 512x512 uint8 = 262 KB, so N=53,971 train+val crops is about **14 GB**,
written in a single pass over the mosaic (one 2048px boundless read per crop, averaged down to
512). Budget roughly an hour; it is I/O bound, not GPU.

**Verify before launching V1/V3.** The build already content-checks 32 random rows against a fresh
live read and exits non-zero if any differ, so a clean build is already verified. To re-check an
existing cache at any time:

```bash
python -m vision_backend.prep_seg_context_cache \
  --manifest $DER/seg_crops_DC_full.csv \
  --imagery  $DER/drg_on_label_grid.tif \
  --out-dir  $DER/seg_context_cache_full \
  --verify-only --verify-samples 64
```

Expect `OK: 64 randomly sampled rows match a live read exactly`. Exit code is non-zero on any
mismatch.

This is deliberately a **content** check, not a row-count one. Rows are addressed by
`rec.index`, so a cache that is merely REORDERED has the right shape and the right count, loads
without complaint, trains without error, and pairs every crop with someone else's surroundings --
producing plausible-but-meaningless numbers. Counting rows cannot detect that; re-reading them can.
Regression-tested both ways in `vision_backend/tests/test_context_cache_alignment.py`.

## 2. Initialize the four wandb sweeps
```bash
cd AI4ExoMars
wandb sweep --project ai4exomars config/variant_v0_sweep.yaml   # big / no-context   -> SWEEP_V0
wandb sweep --project ai4exomars config/variant_v2_sweep.yaml   # small / no-context -> SWEEP_V2
wandb sweep --project ai4exomars config/variant_v1_sweep.yaml   # big / context      -> SWEEP_V1
wandb sweep --project ai4exomars config/variant_v3_sweep.yaml   # small / context    -> SWEEP_V3
```
Record the four sweep IDs (`entity/ai4exomars/<id>`). Each config has `run_cap: 4`.

### 2b. Verify the small/big step-time ratio ON THIS HARDWARE (2 min, do it once)
The params ratio is exact and needs no checking (0.696 / 0.704). The **step-time** ratio could
not be pinned down locally -- an M-series laptop gave ~0.66 with +-12% run-to-run spread, and a
CUDA card at a real batch size has a different profile anyway. Measure it here before spending
four sweeps' worth of GPU time on a claim that may not hold:

```bash
python - <<'EOF'
import statistics, time, torch
from vision_backend.training.builders import build_context_segmentation_model
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
```

If the step ratio lands well outside 0.68-0.72, the width/decoder split needs re-tuning before
launching -- `decoder_channels` is the lever that moves it (it dominates the step), and
`local_base_channels` is the one that moves params. Both live in `run_variant_sweep.sh`'s
`case "$VARIANT"` block and in the header comment of each `config/variant_*_sweep.yaml`.

## 3. Submit runs (4 per variant)
V0 and V2 can start immediately. **V1 and V3 only after Step 1 succeeds.**

```bash
cd ~/AI4ExoMars
export WANDB_API_KEY=...            # the .sub uses getenv = True

# --- V0 and V2 first: neither needs the context cache ---------------------
VARIANT=v0 SWEEP_ID=<entity>/ai4exomars/<SWEEP_V0> condor_submit run_variant_sweep.sub
VARIANT=v2 SWEEP_ID=<entity>/ai4exomars/<SWEEP_V2> condor_submit run_variant_sweep.sub

# --- V1 and V3 ONLY after Step 1 has produced context.npy -----------------
VARIANT=v1 SWEEP_ID=<entity>/ai4exomars/<SWEEP_V1> condor_submit run_variant_sweep.sub
VARIANT=v3 SWEEP_ID=<entity>/ai4exomars/<SWEEP_V3> condor_submit run_variant_sweep.sub
```

Each `.sub` queues **1 agent with `SWEEP_COUNT=4`**, so one submission per variant covers the
`run_cap: 4` budget. `run_variant_sweep.sh` supplies the fixed architecture for the variant it is
given (widths, `--use-context` / `--no-use-context`, `--random-init-encoder`), so the only thing
that changes between these four lines is `VARIANT` and the sweep id.

`VARIANT=v1`/`v3` **hard-fail** with the exact rebuild command if `context.npy` is missing, rather
than silently falling back to the per-item live read -- that fallback exists for a one-batch smoke
and would make a real run crawl.

Without Condor, the same thing directly:
```bash
VARIANT=v0 SWEEP_ID=<entity>/ai4exomars/<SWEEP_V0> ./run_variant_sweep.sh
```

## 4. Collect results
When all runs finish:
```bash
cd ~/AI4ExoMars
python scripts/collect_variant_comparison.py \
  --metrics-dir results/variant_comparison \
  --out         variant_comparison.md
```
Expected: `variant_comparison.md` with best-of-4 per variant — `val/miou` (DC + IG), params, train
step time, inference throughput, peak mem. Confirm **small ≈ 0.70× big** on params and step-time.

## Notes / gotchas
- Order dependency: **context cache (Step 1) → V1/V3 submit.** V0/V2 are independent.
- V1/V3 from-scratch numbers are a **plumbing + relative** read; the context branch is expected to
  be undersold without pretraining (see MODEL_VARIANTS_EXPERIMENT.md §2, §9). Don't treat a weak
  V1/V3 as "context doesn't help."
- If context reads bottleneck throughput, confirm the cache is being used (not the live-read dev
  fallback).
