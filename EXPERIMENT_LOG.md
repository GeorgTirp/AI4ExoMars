# AI4ExoMars — Stage-3 segmentation experiment log

Banked results of the model-design experiments run 2026-09-28 … 2026-10-02 on
the Oxia Planum NOAH-H mosaic. Every number below is reproducible from the
checkpoints, logs and result files listed in each section.

## 1. Common setup (unless a section says otherwise)

| | |
|---|---|
| Labels | NOAH-H **classifier output** (Barrett et al. 2022), Descriptive Classes mosaic `Oxia_NOAHH_Mosaic_DC_4bit25cm_20220203`, decoded to class ids on the label grid (nearest legend colour) |
| Imagery | HiRISE DRG mosaic V1.7, warped onto the label grid (0.241 m/px, `noahh_alignment/`) |
| Crops | 512 × 512 px (124 m); 53,813 train / 1,496 val; 1056-px buffer between train and val (no spatial leakage) |
| Metric | **global** confusion-matrix mIoU over the val set, mean over present classes (13 of 14 — *Boulder fields* has no pixels) |
| Loss | class-weighted cross-entropy (weights from pixel counts, as logged) **+ IG auxiliary head, weight 0.4** (5 Interpretive Groups, built-in `DEFAULT_DC_TO_IG`) — active in **every** run below |
| Optimizer | NAdamW (decoupled WD; norms/biases/rel-pos tables undecayed), cosine schedule with warmup, grad-clip 1.0, EMA 0.9999 (val and saved checkpoint use EMA weights) |
| Schedule | 30 epochs, batch 4, flips (lockstep for context crops), decoder dropout 0.1, drop-path 0 |
| Infra | A100-40GB, `torch.compile`, crop caches on `/fast` |

### Fixes that the comparisons depend on
All runs below were made after these were in place (earlier runs are not comparable):

| fix | effect if absent |
|---|---|
| WD excluded from norms/biases/γ, NAdam decoupled WD | model collapsed to the class prior |
| global confusion-matrix mIoU (was batch-averaged) | non-standard, inflated metric |
| 1056-px train/val buffer | train/val spatial leakage |
| context crops flipped in lockstep with the local crop | context misaligned under augmentation |
| `torch.compiler.disable` on ConvNeXt-V2 GRN | non-finite gradients under compile (SimMIM hybrid) |
| bf16 for the SimMIM hybrid | GRN overflow in fp16 |
| `torch.compile(dynamic=False)` (1c5c76f) | torch 2.5.1 inductor "Failed to find static RBLOCK" crash |
| v0–v3 compile check (diag 17630154) | verified: compiled and eager both skip only the 2 normal GradScaler calibration steps — v0–v3 numbers are trustworthy |

## 2. Model families

| name | encoder | params | step | notes |
|---|---|---|---|---|
| v0 | ConvNeXt-Swin, big (base 52), no context | 28.8 M | 93 ms | ConvNeXt v1 at 1/2–1/8, Swin 2+2+2 at 1/8–1/32 |
| v1 | v0 + context branch (FiLM) | 33.1 M | 100 ms | context = 2048-px crop (494 m) downsampled |
| v2 | ConvNeXt-Swin, small (base 44), no context | 19.7 M | 77 ms | |
| v3 | v2 + context branch (FiLM) | 23.0 M | 84 ms | |
| v3x | v3 with spatial cross-attention fusion | 22.3 M | 82 ms | bottleneck attends to the 16×16 context grid |
| SimMIM hybrid | HybridEncoder ("Model v2") | 31.8 M | 75.5 ms (bf16) | 4×4 patchify stem; ConvNeXt-V2 3+3 at 1/4, 1/8; **Swin ×6 at 1/16; Swin + global attention at 1/32**; 78 % of params in attention matrices |

## 3. Experiments

### E1 — Architecture comparison v0–v3 (wandb `ai4exomars_variant_v{0..3}`)
30 epochs, Bayesian sweep over LR 5e-5…2e-4 and WD 1e-5…1e-2. Clusters 17625133–17625136.

| variant | trials (best val mIoU) | mean |
|---|---|---|
| v0 big | 0.1741 / 0.1809 / 0.1769 | 0.1773 |
| v1 big + context | 0.1784 / 0.1770 | 0.1777 |
| v2 small | 0.1771 / 0.1734 / 0.1707 | 0.1737 |
| v3 small + context | 0.1756 / 0.1769 / 0.1789 | 0.1772 |

A third v1 trial did not complete. Trial-to-trial spread ≈ ±0.004 is the noise floor for single comparisons.
**Learning rate** (peak LR → best mIoU, all 30-epoch trials of E1–E3): the lowest-LR trial was best or within 0.002 of best in 5 of 6 variants; every variant got worse above ~1.2e-4 (e.g. v2@1024: 6.2e-5 → 0.178, 1.45e-4 → 0.168, 1.68e-4 → 0.160). Sweep range since lowered to 1.5e-5…1.2e-4 (1c5c76f).

### E2 — Spatial context fusion, v3x (wandb `ai4exomars_variant_v3_xattn`, sweep g4d3447h)
Cluster 17630151. 0.1712 / 0.1733 / 0.1783, mean **0.1743** (v3 FiLM: 0.1772). No gain; hurt *Continuous + Simple form large ripples* (0.391 vs 0.426).

### E3 — Wider input: v2 at 1024 × 1024 (wandb `ai4exomars_variant_v2_1024`, sweep 3o8ojbc4)
Cluster 17630171; 13,247 train / 377 val crops of 1024 px (same pixel coverage as the 512 set).
Compared on **identical pixels** (the 377 1024-val crops; 512 models on their four non-overlapping 512 quadrants; none of these pixels lies in any 512 training crop) — `results/eval/samepix_1024.json`:

| model | trials | mean |
|---|---|---|
| v2@1024 | 0.1781 / 0.1603 / 0.1675 | **0.169** |
| v2@512 | 0.1798 / 0.1756 / 0.1736 | 0.176 |
| v0@512 | 0.1796 / 0.1838 / 0.1766 | 0.180 |
| SimMIM hybrid@512 | 0.2029 | 0.203 |

The 1024 window is worse, including on every ripple class.

### E4 — SimMIM pretraining A/B on the HybridEncoder (wandb `ai4exomars_pretrain_ab`)
Identical except encoder init; fixed lr 1.107e-4, wd 5.16e-5, seed 42, bf16. Pretrained = `checkpoints/stage1_simmim/last.pt` (1,200 steps × effective batch 256 ≈ 10 passes over 30,675 crops).

| arm | best val mIoU | epoch-30 | cluster |
|---|---|---|---|
| scratch | **0.2003** (ep 26) | 0.2001 | 17630150 |
| pretrained | 0.1937 (ep 19) | 0.1911 | 17630149 |

This short pretraining does not help; the pretrained arm peaks earlier and declines.

### E5 — Muon on the transformer matrices (wandb `ai4exomars_muon`, cluster 17635199)
Scratch arm of E4 with Muon (KellerJordan `SingleDeviceMuon`, pinned f98f1ca, momentum 0.95, Nesterov, 5 Newton-Schulz steps) on the 32 qkv/proj/MLP matrices of the 8 transformer blocks (24.8 M params); everything else NAdamW. LR/WD transferred with Moonlight update-RMS matching (Liu et al. 2025): per-matrix Muon lr 4.3e-4…1.2e-3, per-step decay unchanged (b4e63b0).

| epoch | Muon | NAdamW scratch |
|---|---|---|
| 10 | 0.198 (train loss 1.34) | 0.178 (1.50) |
| 16 | **0.2032** (best) | ~0.190 |
| 30 | 0.194 (0.97) | 0.200 (1.26) |

~2× faster to the same quality, peak +0.003 (within noise), then overfits; +11 % step time (83.9 ms).

### E6 — HetSNGP output layer (wandb `ai4exomars_hetsngp`, cluster 17640120)
Scratch arm of E4 with `decoder.head` replaced by a per-pixel heteroscedastic SNGP layer (Fortuin et al., TMLR 2022, arXiv:2110.02609; e5fafe8): RFF-GP output layer (m = 1024) + rank-6 heteroscedastic logit noise, τ = 1, 32/256 MC samples; Laplace covariance fitted post-training on the EMA weights (13.6 G labelled pixels, 35 min; mean relative posterior variance 2e-6). +17 % step time.

| epoch | 5 | 10 | 15 | 20 | 26 (best) | 30 |
|---|---|---|---|---|---|---|
| HetSNGP | 0.1543 | 0.1746 | 0.1849 | 0.1957 | **0.2007** | 0.2002 |
| plain (E4 scratch) | 0.1620 | 0.1777 | 0.1885 | 0.1950 | 0.2003 | 0.2001 |

Slower start, identical final accuracy; IG head identical throughout. Uncertainty quality: see §4b — underconfident (T* = 0.70) and its predictive entropy does not flag errors; not adopted.

### E7 — Tuning pilot: SimMIM + Muon + Lovász + drop-path, 16 epochs (wandb `ai4exomars_tune_simmim`, sweep jjo7a5f6)
Cluster 17642635 (4 agents × 1 trial; bayes + hyperband). Lovász-softmax (Berman et al., CVPR 2018) added to class-weighted CE; +15 % step time (first wave ran 10× slow from an [N, C] dim-0 sort — fixed, 08f0c7c).

| trial | best | Muon lr | NAdamW lr | Muon wd | NAdamW wd | drop-path | Lovász |
|---|---|---|---|---|---|---|---|
| 2 (3hke8pym) | **0.2055** | 1.9e-4 | 4.8e-5 | 0.013 | 0.027 | 0.10 | 0.63 |
| 3 (v8ppu07f) | 0.2023 | 4.2e-4 | 5.8e-5 | 0.068 | 4e-5 | 0.05 | 0.68 |
| 0 (qs7kgmov) | 0.192 (stopped ep 11) | 2.7e-4 | 8.4e-5 | 0.036 | 1e-4 | 0.19 | 0.54 |
| 1 (ua0040n7) | 0.180 (stopped ep 6) | 6.8e-5 | 2.7e-5 | 3e-5 | 0.007 | 0.12 | 0.30 |

Best model so far at ~half the compute of the 30-epoch runs; the separate contributions of Lovász and drop-path are not isolated.

## 4. Unified per-class evaluation (512 val, all 30-epoch models)
`results/eval/all512_30ep.json` (eval_all512.sub). Mean IoU over trials; ★ = best.

| class | v0 | v1 | v2 | v3 | v3x | SimMIM | SimMIM-pre | SimMIM+Muon |
|---|---|---|---|---|---|---|---|---|
| Rugged bedrock | .535 | .539 | .531 | .540 | .540 | **.558**★ | .554 | .555 |
| Textured non-bedrock | .481 | .481 | .472 | .482 | .480 | .494 | .499 | **.510**★ |
| Continuous + Simple form large ripples | .442 | **.460**★ | .458 | .426 | .391 | .458 | .424 | .449 |
| Fractured bedrock | .290 | .266 | .279 | .290 | .295 | .315 | .314 | **.325**★ |
| Continuous small ripples | .132 | .133 | .123 | .140 | .130 | .146 | .134 | **.149**★ |
| Smooth + Lineated | .111 | .102 | **.132**★ | .097 | .125 | .108 | .130 | .117 |
| Smooth + Featureless | .101 | .087 | .100 | .077 | .094 | .117 | .109 | **.122**★ |
| Textured bedrock | .072 | .079 | .059 | .083 | .073 | .091 | .079 | **.098**★ |
| Isolated + Simple form large ripples | .058 | .069 | .038 | .059 | .062 | **.119**★ | .089 | .079 |
| Rectilinear form large ripples | .042 | .045 | .024 | .055 | .034 | .160 | .159 | **.196**★ |
| Non-bedrock substr. + Non-cont. small ripples | .040 | .044 | .043 | **.050**★ | .039 | .036 | .026 | .036 |
| Bedrock substr. + Non-cont. small ripples | .001 | .004 | .000 | .004 | .001 | .002 | .002 | .004 |
| Smooth bedrock | .000 | .001 | .000 | .001 | .000 | .000 | .000 | .003 |
| **mIoU** | .1773 | .1777 | .1737 | .1772 | .1743 | **.2003** | .1937 | **.2032** |

## 4b. Calibration and uncertainty (half of the 512 val set)
`scripts/eval_uncertainty.py`, `results/eval/uncertainty_512.json`. Val split into interleaved halves: temperature T* fitted on A (Guo et al. 2017), everything reported on B. AUROC = misclassification detection (higher uncertainty on wrong pixels).

| model | mIoU | NLL raw → T* | ECE raw → T* | AUROC entropy / 1−max p | T* |
|---|---|---|---|---|---|
| plain NAdamW (E4) | 0.2030 | 1.162 → 1.141 | 0.075 → 0.041 | 0.667 / 0.683 | 1.17 |
| HetSNGP (E6) | 0.2026 | 1.202 → 1.142 | 0.146 → 0.050 | 0.387 / 0.622 | 0.70 |
| Muon 30 ep (E5) | 0.2063 | 1.160 → 1.124 | 0.089 → 0.033 | 0.697 / 0.705 | 1.24 |
| pilot trial 2 (E7) | **0.2074** | 1.132 → **1.112** | 0.075 → **0.032** | **0.696 / 0.706** | 1.17 |

HetSNGP components alone: GP variance AUROC 0.682 (informative, distance-aware), heteroscedastic variance 0.447 (anti-informative). Plain models are mildly overconfident (T* 1.17–1.24); temperature scaling fixes calibration for free. No OOD test set yet — the GP's main use case is untested.

## 5. Effect sizes and trends

| change | Δ mIoU | |
|---|---|---|
| **SimMIM hybrid vs best ConvNeXt-Swin** | **+0.023** | only effect clearly above noise |
| Muon on transformer matrices | +0.003 peak, ~2× faster | overfits after ep 16 |
| 16-ep Muon + Lovász + drop-path (pilot best) | +0.005 vs E4 | 0.2055, ~half the compute |
| HetSNGP head | +0.0004 | no accuracy change; worse raw calibration |
| width +9 M params (small → big) | +0.004 / +0.0005 | without / with context |
| context branch (FiLM) | +0.0004 / +0.0035 | big / small |
| cross-attention instead of FiLM | −0.003 | |
| 1024 instead of 512 input (same pixels) | −0.007 | |
| short SimMIM pretraining | −0.007 | |

1. **Allocation beats parameter count.** The hybrid has 10 % more params than v0 but +0.023; +9 M params of width gave +0.004.
2. **Depth of within-crop mixing beats a wider view.** Context branch, cross-attention context and the 1024 window all moved mIoU by ≤ 0.007. The hybrid's lead is on pattern-defined classes (rectilinear ripples 0.02–0.055 → 0.16–0.20, isolated simple-form ripples, fractured/textured bedrock) — consistent with its 6 Swin blocks at 1/16 plus global attention at 1/32 (whole-crop receptive field). Which of these drives the gain is not yet isolated.
3. **Fine detail is not the bottleneck** — the hybrid has no 1/2-resolution features.
4. **Context trades classes**: helps ripple subtypes and textured bedrock, hurts the smooth classes (v3 vs v2: Smooth + Lineated −0.035, Continuous simple-form ripples −0.032); net ≈ 0.
5. **Architecture-independent failures**: Smooth bedrock, both Non-continuous small-ripple classes, Smooth + Lineated stay at 0–0.13 in every model.

## 6. Label and data notes (for the achievable ceiling)
- Labels are NOAH-H **predictions**, so mIoU measures agreement with NOAH-H, including its errors; a model that is more correct geologically can score lower.
- Registration DRG ↔ labels: no resolvable offset (separability and edge-energy surfaces flat over ±24 px); residual estimated at a few px (~1 m) — `derived/alignment_report.md`.
- Pixel-scale speckle (60 val crops, 2026-10-02): 2.1 % of pixels differ from their 5×5 majority class; 70 % of connected label regions are ≤ 4 px, but regions ≤ 25 px cover only 0.5 % of the labelled area (≤ 400 px: 3.6 %). Not a grid/tile artifact (boundaries on grid lines at chance rate).

## 7. Open questions / candidate next experiments
- Disentangle the hybrid's gain: (a) global-attention block → Swin, (b) S3 depth 6 → 2 (one run each).
- Second seed of the scratch hybrid (noise level of 0.200).
- Muon with a 16–18-epoch schedule (and drop-path 0.1) — same quality at ~55 % compute.
- Cheap, no training: flip test-time augmentation and an ensemble of the SimMIM-family checkpoints.
- Lovász-softmax loss (direct mIoU surrogate) for the rare classes.
- With the planetary scientists: definitions / possible merging of the near-zero classes; a small expert-labelled val set to measure true accuracy and NOAH-H's own agreement.

## 8. Where things are
| | |
|---|---|
| launchers | `run_variant_sweep.{sh,sub}`, `run_pretrain_ab.{sh,sub}`, `run_muon.{sh,sub}`, `run_hetsngp.{sh,sub}` |
| evaluation | `scripts/eval_global_miou.py` (`--tile auto`, `--json-out`), `eval_samepix.sub`, `eval_all512.sub` |
| result rows | `results/variant_comparison/*.jsonl`, `results/pretrain_ab/*.jsonl`, `results/muon/*.jsonl`, `results/eval/*.json` |
| checkpoints | `checkpoints/variant_v{0,1,2,3,3_xattn,2_1024}/best_30ep_<run>.pt`, `checkpoints/pretrain_ab_{scratch,pretrained}/`, `checkpoints/muon_transformer_scratch/`, `checkpoints/hetsngp_scratch/` |
| logs | `job_outputs/variant_*/agent.<cluster>.<proc>.out`, `job_outputs/{pretrain_ab,muon,hetsngp}/` |
| key commits | 1c5c76f (static compile, LR range, tiled eval), e5fafe8 (HetSNGP), b4e63b0 (Muon routing), 08f0c7c (Lovász, tuning sweep), df3f47c (uncertainty eval) |
