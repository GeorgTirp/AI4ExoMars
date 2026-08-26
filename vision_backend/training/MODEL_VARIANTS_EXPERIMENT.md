# Model-variant comparison — implementation brief (Claude Code, LOCAL machine)

Compare encoder variants for NOAH-H Stage-3 segmentation via short wandb sweeps, holding everything
except the architecture fixed. All variants train **from scratch (no pretraining)** on one unified,
parameterized encoder.

> **Execution model — read first.** This brief is for the **local** coding session. Implement
> **everything** here, run only offline verification, and **launch nothing** (no `wandb sweep`, no
> `wandb agent`, no `condor_submit`, no full cache build). The user pulls the repo on the **server**
> tomorrow; a separate agent runs it from `vision_backend/training/SERVER_LAUNCH.md`, which **this
> session must generate** as its final step (§7).

## Locked decisions (from planning)

1. **One encoder family.** Every variant is `ContextAwareConvNeXtSwinEncoder` (`model_kind=context`)
   with a **context on/off toggle**; no-context variants run it as a single (local) branch. Do not
   use the frozen `HybridEncoder`.
2. **No pretraining.** Stage-3 from random init; no Stage-1/2.
3. **Shrink by width.** "small" = big with narrowed base channels (params ∝ width²), ≈ −30% params
   **and** ≈ −30% time.
4. **Budget.** 4 Bayesian runs per variant, metric `val/miou`.
5. **Context needs a data path (blocker).** The labeled seg loader produces **no context** — only
   the Stage-1 SSL loader does. Context-on variants therefore need a new **context crop cache**
   (C5), built once **on the server** (§7). This splits the work into two phases.

## 1. Variants (2×2)

| id | size | context | phase |
|---|---|---|---|
| **V0** (recommended reference) | big | off | 1 |
| **V1** | big | on | 2 |
| **V2** | small | off | 1 |
| **V3** | small | on | 2 |

## 2. Phase split

- **Phase 1 — V0, V2 (context-off):** needs only C1–C4 and the **existing** seg loader. Fully
  runnable with no data work. This is the clean part: it answers "what does −30% width cost/save?"
- **Phase 2 — V1, V3 (context-on):** additionally needs **C5** (context crop cache + loader path)
  and a one-time cache build on the server.
- **Honest caveat (put in the results):** from-scratch on limited labels **undersells the context
  branch** (its payoff normally needs pretraining). Treat V1/V3 here as a plumbing + relative read,
  not the context verdict; the real context comparison wants the pretrained follow-up (§9).

## 3. Code changes — implement ALL locally (launch nothing)

### C1 — Context-off mode
`ContextAwareSegmentationModel.forward` requires `context_x` (raises on `None`); add a local-only
path. Add `use_context: bool` to `build_context_encoder`/`build_context_segmentation_model` and the
encoder/seg classes; when `False`, don't build/run the context sub-encoder or fusion, and
`forward(local_x, context_x=None)` is valid. Add `--use-context/--no-use-context` to
`train_stage3_segmentation_finetune.py`. **Preserve the `decoder.head` contract** (features.py /
uncertainty / pc_align / mars-inference).
*Test:* `use_context=False` → no context params, runs on `local_x` alone; `True` unchanged.

### C2 — Width-scaled big/small
Dims derive from `local_base_channels` (skip2 ×1, skip4 ×2, skip8 ×4, bottleneck ×16 with
`use_stage32`); params ∝ width². Pick **big** `local_base_channels` so encoder params land near the
current model (~24–31 M — print the param table). **small** = `round(big × 0.84)` for every width
knob (`local_base_channels`, `context_base_channels`, `context_dim`), keeping their ratios.
**Verify + adjust** until small is **0.68–0.72×** big in **both** params and measured train-step
time. Keep depths/heads/`use_stage32`/`decoder_channels=256` identical.

### C3 — From-scratch Stage-3 wiring
Add `--random-init-encoder` (skip checkpoint load; train all params). Force `llrd=1.0` and
`freeze_encoder_epochs=0` (nothing pretrained to preserve). Default `--epochs 50` (configurable).

### C4 — Measurement harness
Per run, log to wandb + a CSV/MD row: `params_encoder`, `params_total`, `val/miou` (DC),
`val/miou_ig`, mean **train step time (ms)**, **inference throughput (img/s)** at 512², **peak GPU
mem (GB)**. Emit `variant_comparison.md` (best-of-4 per variant) — the collector can run after the
server runs finish.

### C5 — Context crop cache + loader path (Phase 2)
**Cache builder (new, server-run — do NOT run locally):** a script (extend `prep_seg_crops.py` /
`prep_seg_crop_cache.py`, or new `prep_seg_context_cache.py`) that, for each crop in the seg
manifest, reads a **boundless 2048² window centered on the crop** from `drg_on_label_grid.tif`
(`_read_live` already uses `boundless=True`), downsamples to `context_output_size`, normalizes
exactly like the local crop, and writes a `context.npy` **index-aligned** to `images.npy`. Expose
the imagery path, manifest, `--context-size 2048`, `--context-output-size`, and output dir as args.
**Loader:** extend `SegmentationCropDataset.__getitem__` to return `batch["context"]` from that
cache when a context backend/flag is set (dev fallback: a live boundless read of the same window, so
a one-batch smoke works without the cache). Extend `parse_segmentation_batch`/loader config so
`context` flows into `model(local, context)`. Keep context-off the default so Phase-1 loading is
untouched.
*Tests:* context tensor shape/normalization; **index alignment** with `images.npy`; context-off path
unchanged; dev live-read fallback yields the same shape as the cache.

## 4. Sweep configs (create locally; DO NOT launch)
Create `config/variant_{v0,v1,v2,v3}_sweep.yaml` from `stage3_segmentation_finetune_sweep.yaml`.
Fixed per file: `--model-kind context`, `--use-context {on|off}`, big/small width flags,
`--random-init-encoder`, `--epochs 50`, crop 512, `--num-classes` + `--num-classes-ig`,
`--decoder-dropout 0.1`, `--ema-decay 0.9999`, `llrd=1.0`. `method: bayes`, `run_cap: 4`,
`metric: val/miou (max)`. **Swept only:** `optimization.learning_rate` (log 1e-4…3e-3),
`optimization.weight_decay` (log 1e-4…1e-1). V1/V3 headers must note they require the context cache.

## 5. Held constant vs swept
Constant across variants: seg dataset (aligned NOAH-H grid), crop 512, epochs, aug (H+V flips, no
rotation), `ignore_index`, `decoder_channels=256`, IG head on, dropout 0.1, EMA on, optimizer
routing. Only `{width, use_context}` vary between variants; only `{lr, weight_decay}` within a
sweep.

## 6. Local acceptance (offline only — no training launched)
- `pytest` green incl. new tests (C1–C5).
- A **one-batch CPU/tiny smoke per variant** (V0–V3) via synthetic or the C5 live-read fallback:
  forward + one backward step runs; **no full training, no full cache build.**
- Param + step-time table printed; small = 0.68–0.72× big on both.
- `decoder.head` is a per-pixel `Conv2d(decoder_channels, num_classes, 1)` in every variant.

## 7. Server handoff — GENERATE `SERVER_LAUNCH.md` (final step, required)
After implementing C1–C5 and the configs, **write `vision_backend/training/SERVER_LAUNCH.md`** and
fill in the **exact** commands (real script names/args/paths as implemented), covering:
1. **The single context-cache build call** (one command) — the user explicitly wants this spelled
   out. Needed only for V1/V3.
2. `wandb sweep` init for all four variant configs (→ 4 sweep IDs).
3. Submitting **4 agent runs per variant** (Condor via the existing `tune_stage3.sh`/`.sub`
   pattern, or `wandb agent`).
4. Running the `variant_comparison.md` collector when runs finish.
Order note in that file: V0/V2 need no cache; **V1/V3 require step 1 first.**
**Do not execute any of these commands in this local session** — only write them into the runbook.

## 8. Guardrails
- **Launch nothing locally** (no sweep/agent/condor/full-cache-build). Implementation + offline
  smoke only.
- Do not modify `HybridEncoder`/`blocks_v2` (frozen contract).
- Do not change dataset/crop/free-lever settings between variants — only width & context differ.
- All new flags default to current behavior; existing Stage-3 runs unaffected.
- Preserve the `decoder.head` per-pixel classifier contract everywhere.

## 9. Optional follow-up (note; do not implement)
If the from-scratch signal is promising, redo the same 4-cell matrix **with pretraining**
(Stage-1 SSL → optional Stage-2 distill for the small ones). The C5 context loader is reused as-is;
that is the comparison that reflects deployed performance and where the context branch can actually
pay off.
