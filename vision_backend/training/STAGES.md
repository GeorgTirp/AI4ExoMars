# Training Stages

This directory holds the shared pieces for a three-stage HiRISE pipeline.

## Stage Files

- `train_stage1_teacher_ssl.py`
  - Large teacher self-supervised pretraining on paired local/context HiRISE crops.
- `train_stage2_student_distill.py`
  - Teacher-to-student unlabeled feature distillation.
- `train_stage3_segmentation_finetune.py`
  - Supervised segmentation fine-tuning on the labeled dataset.

## Shared Modules

- `training/utils.py`
  - Shared path handling, seeding, checkpoint/history saving, loader hooks, and segmentation epoch helpers.
- `training/wandb_utils.py`
  - Common Weights & Biases run and sweep helpers.
- `training/builders.py`
  - Central model builders and checkpoint-loading helpers, including
    `load_segmentation_model_from_checkpoint` -- the one place that turns a saved
    stage-3 checkpoint back into a runnable model (used by the stage-3 script itself,
    both `fit_gaussians.py` and `fit_neural_pca.py` below, and MarsObsLabeling's
    `mars-inference`).
- `training/distillation.py`
  - Teacher-student feature distillation wrapper.
- `training/segmentation.py`
  - Lightweight segmentation decoder and model wrapper.
- `model/features.py`
  - Shared feature-extraction hooks (`extract_pixel_features`, `extract_pooled_features`,
    `get_classifier_weight_vector`) that both post-hoc analyses below hang off of --
    they read `model.decoder.head`'s input, which exists identically on
    `SingleBranchSegmentationModel` and `ContextAwareSegmentationModel` regardless of
    encoder, so neither analysis needs to know encoder internals.

## Post-hoc Analysis: Uncertainty & Explainability

Two analyses over a *trained* segmentation model, both consumed by
MarsObsLabeling's `mars-inference` (Uncertainty Heatmap toggle and the class
Summary window's neural-PCA gallery). Each has a library module (pure
model-hooking + math, unit-tested against the real model classes) and an offline
CLI script that scans a labeled dataset and writes a portable artifact next to
the checkpoint.

- `uncertainty/malahanobis.py` + `uncertainty/uncertainty_mapping.py`
  - Mahalanobis-distance out-of-distribution / epistemic uncertainty (Lee et al.
    2018): fit per-class Gaussians over training-set features, score new pixels by
    distance to the nearest class Gaussian. `uncertainty_mapping.py` also has
    `softmax_confidence_map` / `predictive_entropy_map`, which need no fitting at
    all -- available for any model, trained or not.
  - `uncertainty/fit_gaussians.py` -- offline calibration script; writes
    `<checkpoint_stem>.uncertainty.pt`.
- `pc_align/neural_pca.py`
  - "Neural PCA": class-wise PCA over `psi_k(x) = w_k (elementwise*) phi(x)`
    (classifier weight times pooled features) -- the top components are a class's
    dominant directions of variation in class-relevant feature space; ranking
    samples along each component surfaces the images that most strongly activate
    it. Ported from
    `Maritan-Terrain-Sematic-Segmentation/src/martian_terrain_segmentation/explainability.py`.
  - `pc_align/fit_neural_pca.py` -- offline script; scans labeled crops, fits PCA
    per class, and saves a self-contained gallery of ranked thumbnail crops (no
    need for the training corpus later) to `<checkpoint_stem>.npca.pt`.

Both scripts share the same manifest/imagery/label CLI shape as
`train_stage3_segmentation_finetune.py`'s default NOAH-H loader. Example:

```bash
python3 AI4ExoMars/vision_backend/uncertainty/fit_gaussians.py \
  --checkpoint checkpoints/stage3_segmentation.pt \
  --manifest-path data/seg_crops/manifest.csv \
  --imagery-path data/seg_crops/imagery.tif \
  --label-path data/seg_crops/labels.tif

python3 AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py \
  --checkpoint checkpoints/stage3_segmentation.pt \
  --manifest-path data/seg_crops/manifest.csv \
  --imagery-path data/seg_crops/imagery.tif \
  --label-path data/seg_crops/labels.tif
```

## Sweep Templates

- `training/sweeps/stage1_teacher_ssl.json`
- `training/sweeps/stage2_student_distill.json`
- `training/sweeps/stage3_segmentation_finetune.json`

## Example Commands

Patch extraction:

```bash
python3 AI4ExoMars/vision_backend/hirise_patchloader.py \
  --input-dir data \
  --output-dir data/hirise_context_pairs \
  --patch-size 256 \
  --context-size 2048 \
  --context-output-size 256 \
  --stride 256 \
  --write-context-cache
```

Runtime backends:

- `auto`
  - Prefer per-sample cached `256x256` context tensors when available, otherwise fall back to the shared per-image context base, then to raw JP2 reads.
- `offline_shared_context`
  - Use local patch files plus one shared downsampled context base per source image.
- `offline_paired_context`
  - Use local patch files plus precomputed per-sample context tensors for maximum cluster throughput.
- `online_jp2`
  - Read local and context windows directly from the raw JP2 source image.

Stage 1:

```bash
python3 AI4ExoMars/vision_backend/train_stage1_teacher_ssl.py \
  --index-path data/hirise_context_pairs/patch_index.csv \
  --dataset-backend auto \
  --wandb --wandb-project ai4exomars
```

Stage 2:

```bash
python3 AI4ExoMars/vision_backend/train_stage2_student_distill.py \
  --index-path data/hirise_context_pairs/patch_index.csv \
  --dataset-backend auto \
  --teacher-checkpoint checkpoints/stage1_teacher_ssl.pt \
  --wandb --wandb-project ai4exomars
```

Stage 3:

```bash
python3 AI4ExoMars/vision_backend/train_stage3_segmentation_finetune.py \
  --encoder-checkpoint checkpoints/stage2_student_distill.pt \
  --loader-factory martian_terrain_segmentation.dataloader:create_ai4mars_dataloaders \
  --loader-config-path path/to/segmentation_loader_config.json \
  --wandb --wandb-project ai4exomars
```

## Running A Sweep

```bash
python3 AI4ExoMars/vision_backend/train_stage1_teacher_ssl.py \
  --wandb \
  --wandb-project ai4exomars \
  --wandb-sweep-config AI4ExoMars/vision_backend/training/sweeps/stage1_teacher_ssl.json
```
