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
> Per the user's decision, `decoder_channels` is scaled too (256 -> 176 for the small variants;
> the pre-launch guess was 192, corrected to 176 by the step-2b measurement below).
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

## 0b. POST-MORTEM of the 2026-08-27 launch — read before relaunching

All four agents were held; none produced a single row of `results/variant_comparison/`.
Four separate causes, all now fixed in code — but **step 1a below is a prerequisite that was
never run**, so re-submitting without it repeats the same 36 h for nothing.

| job | variant | held by | cause |
|-----|---------|---------|-------|
| 17488132 | v0 | `periodic_hold` after 36 h | 45 min/epoch; reached epoch 48 of **trial 1 of 4** |
| 17488146 | v2 | `periodic_hold` after 36 h | same |
| 17488195 | v1 | `on_exit_hold`, 39 s | context-cache guard compared cache rows to the *train split* |
| 17488196 | v3 | `on_exit_hold`, 34 s | same |

- **The padded crop cache (§0, `seg_crop_cache_full/`) was never built.** Every crop was read
  live from the DEFLATE-tiled GeoTIFFs, and `--spatial-jitter-px 32` puts each window off the
  tile grid, so up to 4 tiles were decompressed per crop per epoch. `run_variant_sweep.sh` now
  hard-fails if it is missing.
- **`--batch-size` defaulted to 4**, i.e. 13,492 optimizer steps/epoch on an A100-80GB that was
  nowhere near full. The launcher now passes `BATCH_SIZE=16`. *The swept LR range is only
  comparable across runs at the same batch size.*
- **The context guard was wrong**, not the cache: rows are addressed by `rec.index` into the full
  manifest, so a 55,702-row cache against a 53,971-row train split is correct. It is now a bounds
  check (`seg_dataset._check_cache_spans_split`), with a regression test for the train/val case.
- **One job ran all 4 trials** (`SWEEP_COUNT=4`, `queue 1`), so hitting the wall in trial 1 lost
  everything. Now `SWEEP_COUNT=1`, `queue 4`, plus `--per-run-checkpoint` so trials stop
  overwriting each other's `best.pt`.
- Per-batch metrics did ~45 CUDA syncs per training step; now one.

**Still open, and NOT a plumbing issue:** v0's 48 epochs never learned — `val_miou` sat at
0.034227 and `val_miou_ig` at 0.127819, identical to 6 decimals from epoch 2 to 48, while
train_loss moved only 2.359 → 2.301. That is a collapsed model predicting one class, and no
amount of throughput fixes it. Diagnose it on a short run before spending the sweep budget.

## 0c. Relaunching from cold — the whole sequence, in order

Each step is a gate: it is cheap, and it fails loudly. Do not skip ahead to §3 — the last launch
went straight there and spent 4 GPU-days producing nothing. Steps 1-4 cost no GPU time at all.

```bash
cd ~/AI4ExoMars

# 1. Gate: the code changes from the 2026-09-07 session were never executed.
#    Not on the login node -- its process limit kills anything that forks, and
#    test_context_cache_alignment builds its fixture via subprocess.
condor_submit -i           # or any interactive slot, then:
pytest -q vision_backend/tests/

# 2. Gate: the context cache is intact (content check, not a row count).
DER=data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived
python -m vision_backend.prep_seg_context_cache \
  --manifest $DER/seg_crops_DC_full.csv --imagery $DER/drg_on_label_grid.tif \
  --out-dir  $DER/seg_context_cache_full --verify-only --verify-samples 64
# expect: OK: 64 randomly sampled rows match a live read exactly

# 3. Build the padded crop cache (§1a). ~37 GB, I/O bound, no GPU. REQUIRED.
#    chmod is needed once: the file was added by an agent that had no shell.
chmod +x prep_crop_cache.sh
condor_submit prep_crop_cache.sub          # wait for it before step 4

# 4. Gate: does it learn, and does batch 16 fit? Use a THROWAWAY sweep so this
#    does not eat one of the four run_cap trials of a real variant.
wandb sweep --project ai4exomars config/variant_v0_sweep.yaml     # -> SMOKE id
VARIANT=v0 SWEEP_ID=<entity>/ai4exomars/<SMOKE> SWEEP_COUNT=1 EPOCHS=3 \
  ./run_variant_sweep.sh --train-fraction 0.02 --val-fraction-of-split 0.1
```

Step 4 is the one that matters. Read its output for all three of:
- `[stage3] crop cache: ...` — not the WARNING. If you see the warning, step 3 did not land.
- `Train batches:` ~= 0.02 * 53,971 / 16, i.e. batch 16 is actually in effect and it fits.
- **train_loss falling and val_miou moving.** v0's last run held both dead flat for 48 epochs
  (§0b). On 2% of the data for 3 epochs it should visibly overfit; if the loss sits at ~2.3 and
  mIoU never moves off its first value, stop — that is the open bug, and the full sweep will
  reproduce it four times at full price.

Only once step 4 is clean:

```bash
# 5. Four FRESH sweeps. Do not reuse the August ids: v1 (9atwfvnm) and v3
#    (geemh0wd) already spent 3 of their 4 run_cap trials on the crashed runs,
#    and their Bayes priors are seeded with those failures.
wandb sweep --project ai4exomars config/variant_v0_sweep.yaml   # -> SWEEP_V0
wandb sweep --project ai4exomars config/variant_v2_sweep.yaml   # -> SWEEP_V2
wandb sweep --project ai4exomars config/variant_v1_sweep.yaml   # -> SWEEP_V1
wandb sweep --project ai4exomars config/variant_v3_sweep.yaml   # -> SWEEP_V3

# 6. Submit all four. Each line queues 4 jobs of 1 trial (§3).
export WANDB_API_KEY=...
for V in v0 v1 v2 v3; do
  VARIANT=$V SWEEP_ID=<entity>/ai4exomars/<SWEEP_$V> condor_submit run_variant_sweep.sub
done

# 7. After the FIRST job reports an epoch, check the new epoch time against
#    MaxTime=36h before letting the other 15 run long. 45 min/epoch was the
#    live-read figure; if it has not dropped sharply, the cache is not being used.
condor_q -batch
```

Then §4 to collect. Total budget: 16 jobs x 1 trial x 50 epochs.

## 1a. Build the padded crop cache — REQUIRED for all four variants
Run once. `run_variant_sweep.sh` refuses to launch without it.

```bash
chmod +x prep_crop_cache.sh     # once
condor_submit prep_crop_cache.sub
```

Same thing directly, if you would rather not queue it:

```bash
cd ~/AI4ExoMars
DER=data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived

python -m vision_backend.prep_seg_crop_cache \
  --manifest       $DER/seg_crops_DC_full.csv \
  --imagery        $DER/drg_on_label_grid.tif \
  --labels         $DER/labels_DC_classid.tif \
  --jitter-margin  32 \
  --out-dir        $DER/seg_crop_cache_full
```

**Size:** 55,702 crops padded to 576x576, two uint8 arrays = **~37 GB**. I/O bound, one pass.
`--jitter-margin` must be >= the run's `--spatial-jitter-px` (32), or the loader refuses the cache.

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

**Size/time:** one row is 512x512 uint8 = 262 KB, so N=55,702 manifest rows (53,971 train +
1,731 val — the cache spans the WHOLE manifest, not a split) is about **15 GB**,
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

### 2b-RESULT (measured 2026-08-27, this cluster) -- the gate tripped, widths were re-tuned

The pre-launch `decoder_channels=192` measured **0.758x** step on an A100-80GB and **0.762x** on an
A100-40GB: params was exactly on target (0.696) but the step ratio was out of band on both cards.
`calib_variant_widths.sh` then gridded `local_base_channels` x `decoder_channels`
(44-50 x 144-192, big baseline re-measured in the same job). Exactly one point had BOTH ratios
inside 0.68-0.72:

| lbc | cbc | cdim | dch | params | p_ratio | t_ratio |
|-----|-----|------|-----|--------|---------|---------|
| 44 | 22 | 217 | **176** | 19,729,234 | **0.684** | **0.718** | <- adopted
| 44 | 22 | 217 | 192 | 20,082,002 | 0.696 | 0.762 | (old guess, out of band)
| 46 | 23 | 226 | 160 | 21,057,564 | 0.730 | 0.699 | (best step, params out of band)

So the small variants are now `44 / 22 / 217 / 176`. Both adopted ratios sit near their band edges
(params low, step high) -- that is inherent, not slack: raising `local_base_channels` to centre
params pushes the step ratio straight back out, which is what the 46-row above shows.
Re-run `calib_variant_widths.sub` if the architecture or the GPU model changes.

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

Each `.sub` queues **4 jobs of `SWEEP_COUNT=1`**, so one submission per variant still covers the
`run_cap: 4` budget, but a job that is held or evicted costs one trial rather than all four (see
§0b). `run_variant_sweep.sh` supplies the fixed architecture for the variant it is
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
