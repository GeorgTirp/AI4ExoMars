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
  - Central model builders and checkpoint-loading helpers.
- `training/distillation.py`
  - Teacher-student feature distillation wrapper.
- `training/segmentation.py`
  - Lightweight segmentation decoder and model wrapper.

## Sweep Templates

- `training/sweeps/stage1_teacher_ssl.json`
- `training/sweeps/stage2_student_distill.json`
- `training/sweeps/stage3_segmentation_finetune.json`

## Example Commands

Stage 1:

```bash
python3 AI4ExoMars/vision_backend/train_stage1_teacher_ssl.py \
  --index-path data/hirise_context_pairs/patch_index.csv \
  --wandb --wandb-project ai4exomars
```

Stage 2:

```bash
python3 AI4ExoMars/vision_backend/train_stage2_student_distill.py \
  --index-path data/hirise_context_pairs/patch_index.csv \
  --teacher-checkpoint checkpoints/stage1_teacher_ssl.pt \
  --wandb --wandb-project ai4exomars
```

Stage 3:

```bash
python3 AI4ExoMars/vision_backend/train_stage3_segmentation_finetune.py \
  --encoder-checkpoint checkpoints/stage2_student_distill.pt \
  --loader-factory martian_terrain_segmentation.dataloader:create_ai4mars_dataloaders \
  --loader-config-path path/to/segmentation_loader_config.json \
  --num-classes 5 \
  --wandb --wandb-project ai4exomars
```

## Running A Sweep

```bash
python3 AI4ExoMars/vision_backend/train_stage1_teacher_ssl.py \
  --wandb \
  --wandb-project ai4exomars \
  --wandb-sweep-config AI4ExoMars/vision_backend/training/sweeps/stage1_teacher_ssl.json
```
