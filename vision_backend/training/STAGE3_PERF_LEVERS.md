# Stage-3 performance levers — implementation brief (Claude Code)

> **Status: F1–F6 implemented.** All six free levers are wired and default-off
> (neutral defaults reproduce the previous behavior exactly). M1 (ASPP) and M2
> (context branch) are **not** implemented — still decide-later.
>
> | Lever | Flag | Default | Suggested |
> |---|---|---|---|
> | F1 IG aux head | `--ig-loss-weight` (+`--num-classes-ig`) | `0.0` (off) | `0.4` |
> | F2 long-tail loss | `--loss` / `--logit-adjust-tau` | `ce` / `1.0` | `logit_adjusted` |
> | F3 layer-wise LR decay | `--llrd` | `1.0` (off) | `0.7`–`0.8` |
> | F4 decoder dropout | `--decoder-dropout` | `0.0` (off) | `0.1` |
> | F5 EMA eval + checkpoint | `--ema-decay` | `0.0` (off) | `0.9999` |
> | F6 flips, no rotation | — | already correct | verified by test |
>
> New modules: `training/hierarchy.py` (DC→IG), `training/losses.py`.
> Tests: `tests/test_stage3_levers.py` (32 tests).
> One cross-repo fix was required: MarsObsLabeling's `_forward_logits` now
> unwraps the `{"dc","ig"}` dict an IG-trained checkpoint returns.

Implement the **free** performance levers below on the existing Stage-3 segmentation
fine-tuning path. These are decoder/loss/optimizer/eval-side changes only — **do not touch the
frozen encoder** (`model/hybrid_encoder.py`, `model/blocks_v2.py`). A separate "moderate"
section (ASPP, context branch) is included at the end but is **optional / decide-later**.

## Global guardrails (apply to every change)

1. **Preserve the `decoder.head` contract.** `model/features.py` (`get_classifier_head`) and the
   whole post-hoc stack — `uncertainty/malahanobis.py`, `pc_align/neural_pca.py`, and
   MarsObsLabeling `mars-inference` — require `model.decoder.head` to remain a
   `nn.Conv2d(decoder_channels, num_classes, 1)` applied to a per-pixel feature map at input
   resolution. Add new heads/modules as *separate* attributes; never rename or repurpose `.head`.
2. **Neutral defaults = current behavior.** Every new flag defaults so the model trains exactly as
   today (IG weight 0, `loss=ce`, `llrd=1.0`, dropout 0, `ema_decay=0`). This guarantees no
   regression and lets each lever be A/B-tested via the existing wandb sweep.
3. **Keep `ignore_index` semantics** everywhere (partial labels; no background class).
4. **Resume-safe.** Any new optimizer/EMA state must save and restore in the Stage-3 checkpoint.
5. Add a unit test per lever under `vision_backend/tests/` and keep the existing suite green.

---

## FREE levers (implement all)

### F1 — Hierarchical Interpretive-Group (IG) auxiliary head  ★ highest value

**Why:** NOAH-H's 14 descriptive classes (DC) roll up to 5 interpretive groups (IG); the paper
finds IG scores higher and the two levels are complementary. A coarse aux head regularizes the
fine classes through the hierarchy and rescues rare DC classes. Free (one extra 1×1 head + a loss
term on shared features).

**Where:** `training/segmentation.py` (`LightweightSegmentationDecoder`,
`SingleBranchSegmentationModel`), loss in `training/utils.py:run_segmentation_epoch`.

**Change:**
- In `LightweightSegmentationDecoder.__init__`, add `self.head_ig = nn.Conv2d(decoder_channels,
  num_classes_ig, 1)`. Keep `self.head` (DC) exactly as is.
- In `forward`, return both DC and IG logits from the same final feature map `x`
  (e.g. return a dict `{"dc": self.head(x), "ig": self.head_ig(x)}`, or add a parallel return).
  Update `SingleBranch`/`ContextAware` wrappers to pass both through. **`decoder.head` must still be
  invoked on the input feature map so the pre-classifier hook in `features.py` is unchanged.**
- Loss: `total = CE_dc(logits_dc, y_dc) + ig_loss_weight * CE_ig(logits_ig, y_ig)`, both with
  `ignore_index`.
- IG targets: derive from DC targets with a fixed `dc_to_ig` LongTensor `[num_classes_dc]`
  (map on the fly; `ignore_index` maps to `ignore_index`). Build/validate it against
  `noahh_alignment` `classes_DC.json` / `classes_IG.json`. Expected mapping (NOAH-H taxonomy):

  | DC class | IG group |
  |---|---|
  | Smooth+Featureless, Smooth+Lineated, Textured non-bedrock | Non-bedrock |
  | Smooth / Textured / Rugged / Fractured bedrock | Bedrock |
  | Continuous+Simple, Isolated+Simple, Rectilinear large ripples | Large ripples |
  | Continuous small, Bedrock-substrate & Non-bedrock-substrate non-continuous small ripples | Small ripples |
  | Boulder fields | Other cover |

**Flags:** `--ig-loss-weight` (default **0.0** = off; suggested on-value 0.4), `--num-classes-ig`
(infer from `classes_IG.json` if absent).

**Optional (cleaner, note only):** instead of a separate head, compute IG probabilities by summing
DC softmax over group membership (marginalization) → guarantees DC/IG consistency, no new params.
Prefer the separate head for v1 (matches the paper treating them as complementary).

**Tests:** forward returns DC+IG logits at input resolution; `dc_to_ig` maps every DC id and
`ignore_index`; `ig_loss_weight=0` reproduces current loss exactly; small, bounded param increase.

**DoD:** val logging reports both DC mIoU and IG mIoU; with the aux head on, IG mIoU ≥ DC mIoU.

---

### F2 — Logit-adjusted / balanced-softmax loss (long-tail)

**Why:** NOAH-H reports naive class balancing helped boosted classes but hurt others; logit
adjustment is a cleaner long-tail fix than inverse-frequency reweighting. Free (loss-only).

**Where:** `training/utils.py:run_segmentation_epoch` (currently
`nn.CrossEntropyLoss(ignore_index=...)`).

**Change:** make the loss pluggable. Add balanced-softmax / logit-adjusted CE: add
`tau * log(prior_c)` to the logits before softmax, where `prior_c` = training-set pixel frequency
of class `c` (computed once from the manifest/labels, excluding `ignore_index`). Apply the same to
the IG head with IG priors.

**Flags:** `--loss {ce,balanced_softmax,logit_adjusted}` (default **ce**), `--logit-adjust-tau`
(default 1.0). **Do not stack with existing class-weights** — if `loss!=ce`, disable/ignore the
class-weight vector and log a warning.

**Tests:** with uniform priors, balanced_softmax == plain CE (within fp tol); priors computed
correctly and exclude ignore; gradients finite.

---

### F3 — Layer-wise LR decay (LLRD) for fine-tuning

**Why:** LLRD (smaller LR toward the stem) is the standard, reliable way to fine-tune a pretrained
encoder and generally beats long encoder freezes. Free (per-group LR multipliers).

**Where:** `model/optimizers.py` (`split_decay_param_groups`, `create_optimizer`,
`build_routed_muon_nadam_optimizer`).

**Change:** extend param grouping with a depth-indexed LR scale. Assign encoder components a depth
rank and multiply each group's base LR by `llrd ** (max_depth - rank)`; decoder + heads get full
LR (rank = top). Suggested rank order (shallow→deep multiplier): `stem < s1 < down1 < s2 < down2 <
s3 < down3 < s4/norm < decoder/heads`. Must compose with both the AdamW path and the routed
Muon+NAdam path (scale within each existing group).

**Flags:** `--llrd` (default **1.0** = current behavior; suggested 0.7–0.8). Complements, and can
replace, `freeze_encoder_epochs`.

**Tests:** `llrd=1.0` reproduces today's groups/LRs exactly; `llrd<1` yields monotonically
decreasing LR from head→stem; every parameter lands in exactly one group.

---

### F4 — Dropout in the decoder head

**Why:** In NOAH-H's own overfitting stack (dropout, weight decay, aug, pretraining, parameter
sharing); cheap regularization for the few-label regime.

**Where:** `training/segmentation.py:LightweightSegmentationDecoder`.

**Change:** insert `nn.Dropout2d(p)` immediately before `self.head` (and optionally inside the
`fuse*` blocks). Because dropout is identity at eval, the `features.py` pre-`head` hook and all
uncertainty/PCA/inference analyses are unaffected.

**Flags:** `--decoder-dropout` (default **0.0**; suggested 0.1).

**Tests:** dropout active in train, identity in eval; `decoder.head` still hookable; `p=0`
reproduces current outputs.

---

### F5 — Evaluate & checkpoint the EMA weights

**Why:** Weight EMA gives a free, more stable model for validation/inference; you already have
`training/ema.py` (used in Stage 1). <1% per-step cost.

**Where:** `train_stage3_segmentation_finetune.py` (+ `training/ema.py`, `training/utils.py`).

**Change:** maintain an EMA of the model weights during Stage-3; compute `val/miou` (DC and IG) on
the **EMA** weights; write the EMA as the primary segmentation checkpoint. Ensure EMA state is
saved and restored on resume.

**Flags:** `--ema-decay` (default **0.0** = off; suggested 0.9999).

**Tests:** EMA updates each step; eval path uses EMA; resume restores EMA bit-exactly (test mode).

---

### F6 — Confirm augmentation: H+V flips, NO rotation

**Why:** Free label-consistent aug; rotation is forbidden (sun-azimuth/shading guardrail).

**Where:** the Stage-3 loader (default NOAH-H loader / external
`martian_terrain_segmentation.dataloader`). This is a **verify** item — do not add if already
present.

**Change:** confirm horizontal + vertical flips are applied jointly to image and label, and that
**no rotation** and no photometric jitter are enabled. If the loader is external and not
configurable here, document the required aug config rather than modifying it.

**Tests:** a flip is applied identically to image and mask; no rotation transform present.

---

## MODERATE levers (optional — decide later; ASPP recommended over the branch)

> Recommendation: if you run one moderate experiment, make it **M1 (ASPP)** — cheaper, single-branch,
> and the exact NOAH-H/DeepLab mechanism for insufficient field-of-view. Treat **M2 (context
> branch)** as conditional: only after an error analysis shows genuinely context-limited mistakes
> *and* ASPP + inference-at-larger-resolution don't close them, noting it reintroduces the
> two-branch design the Model v2 plan removed.

### M1 — ASPP / dilated-context module at the `s4` bottleneck  (recommended moderate)

**Why:** Expands effective receptive field within a 512 crop — NOAH-H's stated fix for framelet
FOV. Cheap because it operates on the stride-32 map.

**Where:** `training/segmentation.py`, between `features["s4"]` and `decoder.bottleneck_proj`.

**Change:** add an ASPP module — parallel dilated 3×3 convs at rates e.g. `[1,6,12,18]` + an
image-pool branch, concatenated and projected to `bottleneck_channels`/`decoder_channels`. Feeds
the existing decoder; `decoder.head` unchanged.

**Flags:** `--use-aspp` (default **off**), `--aspp-rates` (default `6,12,18`).

**Tests:** shapes preserved into the decoder; `--use-aspp` off ⇒ identical to current model.

### M2 — Context branch  (conditional; only if error analysis justifies)

**Why:** Injects km-scale context via the existing `ContextAwareSegmentationModel` +
`hirise_patchloader` paired local/context crops. Moderate cost (extra, lighter encoder forward).

**Where:** wire Stage-3 to the context path (`ContextAwareSegmentationModel`, paired-crop loader).

**Change:** add a switch selecting single-branch vs context-aware Stage-3; when on, feed
`(local, context)` pairs and use the context model. Keep `decoder.head` contract intact.

**Flags:** `--use-context-branch` (default **off**).

**Caveats to record in the PR:** reintroduces the two-branch design removed in the Model v2 plan;
compare against M1 + infer-at-larger-resolution before adopting.

**Tests:** context path runs end-to-end on a paired-crop batch; single-branch path unchanged when
off.

---

## Suggested wandb sweep additions (after implementation)

Add as axes to `config/stage3_segmentation_finetune_sweep.yaml`: `ig_loss_weight`
(0 / 0.3 / 0.5), `loss` (ce / logit_adjusted), `llrd` (1.0 / 0.8 / 0.7), `decoder_dropout`
(0 / 0.1), `ema_decay` (0 / 0.9999). Keep `val/miou` as the metric (log both DC and IG).
