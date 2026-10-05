#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import json
try:
    from vision_backend.training.wandb_utils import (
        add_wandb_arguments,
        finish_wandb_run,
        init_wandb_run,
        log_metrics,
        maybe_run_sweep,
        merge_wandb_config,
    )
except ModuleNotFoundError:
    from training.wandb_utils import (
        add_wandb_arguments,
        finish_wandb_run,
        init_wandb_run,
        log_metrics,
        maybe_run_sweep,
        merge_wandb_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: supervised segmentation fine-tuning of the distilled encoder."
    )
    parser.add_argument(
        "--model-kind",
        choices=("simmim", "context"),
        default="simmim",
        help="Encoder to fine-tune: 'simmim' single-branch HybridEncoder "
             "(default) or the legacy two-branch 'context' model.",
    )
    parser.add_argument(
        "--loader-factory",
        default="seg_dataset:create_segmentation_dataloaders",
        help="Import path to a loader factory returning train/val loaders. "
             "Default = NOAH-H paired (imagery, class-label) crops.",
    )
    parser.add_argument(
        "--loader-config-path",
        default=None,
        help="JSON config forwarded to the loader factory (for the default "
             "NOAH-H loader: manifest_path, imagery_path, label_path, num_classes).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch size = "
             "batch_size * accum_steps; use to reproduce a larger swept "
             "batch size under tighter GPU memory without changing LRs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="LR for the single-optimizer path (use_muon off).")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    # Routed Muon(2-D weights)+NAdam(rest) path -- active with --use-muon.
    # Muon and NAdam get independent LRs and momenta so the sweep can tune each.
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=1e-2)
    parser.add_argument("--nadam-lr", type=float, default=None,
                        help="NAdamW LR for the non-Muon params (default: --learning-rate).")
    parser.add_argument("--nadam-weight-decay", type=float, default=None,
                        help="Decoupled WD for NAdamW's weights (default: --weight-decay).")
    parser.add_argument(
        "--muon-lr-mode", choices=("absolute", "match_adam"), default="absolute",
        help="absolute: --muon-lr is SingleDeviceMuon's own LR (default 0.02). "
             "match_adam: --muon-lr / --muon-weight-decay are AdamW-equivalent "
             "values, converted per matrix by Moonlight's 0.2*sqrt(max(A,B)) "
             "update-RMS matching (Liu et al. 2025).",
    )
    parser.add_argument("--nadam-beta1", type=float, default=0.9)
    parser.add_argument("--nadam-beta2", type=float, default=0.999)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--ignore-index", type=int, default=255,
                        help="Label id excluded from loss/metrics (NOAH-H "
                             "nodata/boundary = 255).")
    parser.add_argument(
        "--lovasz-weight", type=float, default=0.0,
        help="Add lovasz_weight x Lovász-softmax (Berman et al. 2018; a convex "
             "per-class IoU surrogate, every class weighted equally) to the DC "
             "training loss. 0.0 (default) = unchanged. Validation loss stays "
             "plain CE so it remains comparable across runs.",
    )
    parser.add_argument(
        "--loss-kind",
        choices=("ce", "focal", "balanced_softmax", "logit_adjusted"),
        default="ce",
        help="Training loss. 'focal' (Lin et al. 2017) down-weights already-"
             "confident (typically majority-class) pixels and concentrates "
             "gradient on hard ones -- combines with --class-weight-scheme "
             "the same way weighted CE does. 'balanced_softmax' (Ren et al. "
             "2020) / 'logit_adjusted' (Menon et al. 2021) instead add "
             "tau*log(prior_c) to the logits before softmax (see "
             "--logit-adjust-tau) -- a cleaner long-tail fix than inverse-"
             "frequency reweighting per the NOAH-H paper's own observation "
             "that naive class balancing helped some classes but hurt others; "
             "does NOT stack with --class-weight-scheme (ignored with a "
             "warning if both are set). Val loss always uses plain CE "
             "regardless, so val_loss stays comparable across runs.",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss focusing parameter (only used with --loss-kind focal). "
             "0 reduces to (optionally weighted) CE; 2.0 is the paper's default.",
    )
    parser.add_argument(
        "--logit-adjust-tau", type=float, default=1.0,
        help="Strength of the balanced_softmax/logit_adjusted logit adjustment "
             "(only used with those --loss-kind choices). 1.0 = the "
             "textbook balanced-softmax adjustment.",
    )
    parser.add_argument(
        "--ig-loss-weight", type=float, default=0.0,
        help="Weight on an auxiliary Interpretive-Group (IG, 5-class NOAH-H "
             "rollup of the 14 DC classes) CE loss added to the DC loss: "
             "total = CE_dc + ig_loss_weight * CE_ig. 0.0 (default) disables "
             "the aux head entirely (no params added, no behavior change). "
             "The paper reports IG scores higher than DC and the two levels "
             "are complementary; the coarse aux head regularizes the fine "
             "classes through the hierarchy. Suggested on-value: 0.4.",
    )
    parser.add_argument(
        "--num-classes-ig", type=int, default=None,
        help="IG class count (only used with --ig-loss-weight > 0). Defaults "
             "to inferring from --dc-to-ig-path (or the built-in NOAH-H "
             "taxonomy table if that's also unset) -- 5 for the standard "
             "DC->IG rollup.",
    )
    parser.add_argument(
        "--dc-to-ig-path", default=None,
        help="JSON file with a {\"dc_to_ig\": {dc_id: ig_id}} mapping (only "
             "used with --ig-loss-weight > 0), e.g. data/.../derived/"
             "dc_to_ig.json. Defaults to the built-in NOAH-H taxonomy table "
             "(training/hierarchy.py:DEFAULT_DC_TO_IG) if unset.",
    )
    parser.add_argument(
        "--class-weight-scheme",
        choices=("none", "inverse_sqrt", "inverse", "effective_number"),
        default="inverse_sqrt",
        help="Per-class CrossEntropyLoss weighting from training-split pixel "
             "counts (applied to the train loss only, not val). 'none' "
             "reproduces the original unweighted behavior.",
    )
    parser.add_argument(
        "--class-weight-clip-max", type=float, default=10.0,
        help="Cap the max/min weight ratio after normalization, so a handful "
             "of near-empty classes can't destabilize training with huge "
             "gradients (see training.utils.compute_class_weights).",
    )
    parser.add_argument(
        "--spatial-jitter-px", type=int, default=32,
        help="Random +/- translation (pixels) applied to each training crop "
             "before reading, for effective data diversity from a fixed "
             "manifest. 0 disables.",
    )
    parser.add_argument(
        "--brightness-jitter", type=float, default=0.15,
        help="Random brightness jitter fraction on training crops (valid "
             "pixels only). 0 disables.",
    )
    parser.add_argument(
        "--contrast-jitter", type=float, default=0.15,
        help="Random contrast jitter fraction on training crops (valid "
             "pixels only). 0 disables.",
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0,
        help="Clip total gradient norm before each optimizer step (0 disables). "
             "The swept LRs were tuned on the 208-crop AOI manifest (26 steps/"
             "epoch); on the full manifest (6,747 steps/epoch) the same peak LR "
             "drove weights to NaN at the end of warmup with nothing to bound "
             "the update.",
    )
    parser.add_argument(
        "--allow-nonfinite-loss", action="store_true",
        help="Keep training after a nan/inf loss instead of aborting. Off by "
             "default: once weights go non-finite the run can never recover or "
             "checkpoint again, so continuing only wastes GPU time.",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="torch.compile the model. Measured 2.1x faster here (18.8 -> 39.9 "
             "img/s on an RTX 3070) for a one-off ~70s compile.",
    )
    parser.add_argument(
        "--amp-dtype", choices=("fp16", "bf16"), default="fp16",
        help="Mixed-precision dtype on CUDA. fp16 (default) is the historical "
             "setting. Use bf16 for any encoder built from ConvNeXt-V2 blocks -- "
             "i.e. the SimMIM HybridEncoder: GRN takes an L2 norm over the whole "
             "spatial map, a sum of squares that overflows fp16's 65,504 ceiling "
             "on ordinary batches and turns the loss NaN. The HybridEncoder was "
             "also PRETRAINED in bf16. Both A/B arms diverged in fp16 on nearly "
             "the same batch at lr 4.5e-5.",
    )
    parser.add_argument(
        "--channels-last", action="store_true",
        help="Use channels_last memory format (~1.17x here).",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="Train on this fraction of the train split (1.0 = all). Applied "
             "in memory, so the padded crop cache stays valid. Use to make a "
             "hyperparameter sweep affordable -- but note it lowers steps/epoch, "
             "the very quantity whose AOI->full mismatch caused the NaN "
             "divergence, so re-check the winning LR at 1.0 before a long run.",
    )
    parser.add_argument(
        "--val-fraction-of-split", type=float, default=1.0,
        help="Evaluate on this fraction of the val split (1.0 = all). Trims "
             "per-epoch validation cost during sweeps.",
    )
    parser.add_argument(
        "--llrd", type=float, default=1.0,
        help="Layer-wise LR decay: each encoder param group's LR is scaled by "
             "llrd**(depth_from_head), shallower (stem) components discounted "
             "more; decoder/heads always keep the full LR. 1.0 (default) = "
             "every param gets the same LR, today's behavior exactly. "
             "Complements, and can replace, --freeze-encoder-epochs. "
             "Suggested on-value: 0.7-0.8.",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="Exponential moving average of the model weights, updated once "
             "per optimizer step. 0.0 (default) disables EMA entirely -- "
             "current behavior. When > 0, val/miou is computed on the EMA "
             "weights (not the live weights) and the EMA is what gets saved "
             "as the checkpoint. Suggested on-value: 0.9999.",
    )
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Pretrained encoder to initialise from. Not required "
                             "with --random-init-encoder.")
    parser.add_argument(
        "--context-cache-dir", default=None,
        help="Context crop cache from prep_seg_context_cache.py, index-aligned "
             "with the seg manifest. Used with --use-context; without it the "
             "loader falls back to a live per-item read (dev smoke only).",
    )
    parser.add_argument(
        "--crop-cache-dir", default=None,
        help="Padded (imagery, label) crop cache from prep_seg_crop_cache.py, "
             "index-aligned with the seg manifest. Without it the loader reads "
             "every crop live from the DEFLATE-tiled GeoTIFFs, and spatial "
             "jitter puts each window off the tile grid, so up to 4 tiles are "
             "decompressed per crop per epoch. Overrides `cache_dir` in "
             "--loader-config-path.",
    )
    parser.add_argument(
        "--variant-id", default=None,
        help="Label for the model-variant comparison (v0..v3); recorded in the "
             "metrics row and logged to wandb.",
    )
    parser.add_argument(
        "--variant-metrics-path", default=None,
        help="Append this run's params/throughput/memory measurements as a JSON "
             "line here, for collect_variant_comparison.py.",
    )
    parser.add_argument(
        "--random-init-encoder", action="store_true",
        help="Train the encoder from random init: skip the checkpoint load "
             "entirely. Also forces llrd=1.0 and freeze_encoder_epochs=0, since "
             "both exist to protect pretrained weights and there are none.",
    )
    parser.add_argument(
        "--use-context", dest="use_context", action="store_true", default=None,
        help="Run the context-aware encoder with its context branch (model-kind "
             "context only). Requires the loader to supply context crops.",
    )
    parser.add_argument(
        "--no-use-context", dest="use_context", action="store_false",
        help="Run the context-aware encoder as a single (local) branch: no "
             "context sub-encoder, no FiLM conditioning, no context crops needed.",
    )
    parser.add_argument(
        "--context-fusion", choices=("film", "xattn"), default="film",
        help="How the context branch conditions the local branch (model-kind "
             "context with --use-context). 'film' (default): pooled to one global "
             "vector that FiLM-modulates the local maps -- scene-level, no spatial "
             "correspondence. 'xattn': the context grid is kept and the local "
             "bottleneck cross-attends to it, with positions in a shared frame, so "
             "the model knows WHERE in the 494 m window a pattern sits.",
    )
    parser.add_argument("--strict-checkpoint-load", action="store_true")
    # simmim (single-branch HybridEncoder) hyperparameters
    parser.add_argument("--global-base-grid", type=int, default=32)
    parser.add_argument(
        "--hybrid-s3-depth", type=int, default=6,
        help="Architecture ablation (simmim HybridEncoder): number of Swin blocks "
             "at 1/16 resolution. 6 = the frozen layout.",
    )
    parser.add_argument(
        "--hybrid-s4-block", choices=("global", "swin"), default="global",
        help="Architecture ablation: the second S4 block (1/32) is global "
             "attention (frozen layout) or a shifted-window Swin block.",
    )
    # legacy context-model hyperparameters (only used with --model-kind context)
    parser.add_argument("--local-base-channels", type=int, default=32)
    parser.add_argument("--context-base-channels", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--decoder-channels", type=int, default=192)
    parser.add_argument(
        "--decoder-dropout", type=float, default=0.0,
        help="nn.Dropout2d(p) immediately before the decoder's classifier "
             "head(s). 0.0 (default) = identity, current behavior. Identity "
             "at eval regardless, so it never affects inference/uncertainty/"
             "PCA. Suggested on-value: 0.1.",
    )
    parser.add_argument(
        "--use-aspp", action="store_true",
        help="M1: Atrous Spatial Pyramid Pooling (parallel dilated convs + an "
             "image-pool branch) at the encoder bottleneck, before "
             "decoder.bottleneck_proj -- widens the effective receptive field "
             "within a crop without added stride. Off by default (no params "
             "added, no behavior change). Feeds the existing decoder; "
             "decoder.head is unaffected either way.",
    )
    parser.add_argument(
        "--aspp-rates", nargs="+", type=int, default=(6, 12, 18),
        help="Dilation rates for --use-aspp's parallel branches.",
    )
    parser.add_argument(
        "--uncertainty-head", choices=("none", "het", "sngp", "hetsngp"), default="none",
        help="Replace decoder.head with a HetSNGP output layer (model/hetsngp.py; "
             "Fortuin et al., TMLR 2022): 'het' = heteroscedastic low-rank logit "
             "noise (Collier et al. 2021), 'sngp' = random-feature GP output layer "
             "(Liu et al. 2020), 'hetsngp' = both. GP heads need a post-training "
             "Laplace fit (scripts/fit_hetsngp_covariance.py) before their "
             "predictive variance is used.",
    )
    parser.add_argument("--het-num-factors", type=int, default=6,
                        help="Rank R of the heteroscedastic covariance (paper: 6 CIFAR, 15 ImageNet).")
    parser.add_argument("--het-temperature", type=float, default=1.0,
                        help="Softmax temperature tau (paper ablation: ~1.0 works well).")
    parser.add_argument("--het-train-mc-samples", type=int, default=32)
    parser.add_argument("--het-test-mc-samples", type=int, default=256)
    parser.add_argument("--gp-num-inducing", type=int, default=1024,
                        help="Random Fourier feature dimension m.")
    parser.add_argument("--gp-kernel-scale", type=float, default=1.0,
                        help="RBF kernel scale (inputs are divided by its square root).")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument(
        "--swin-depths",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        metavar=("S8", "S16", "S32"),
    )
    parser.add_argument(
        "--swin-num-heads",
        nargs=3,
        type=int,
        default=(4, 8, 16),
        metavar=("H8", "H16", "H32"),
    )
    parser.add_argument("--disable-stage32", action="store_true")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument(
        "--muon-scope", choices=("matrix", "all", "transformer"), default="matrix",
        help="Which weights Muon optimizes. 'transformer' = only the 2-D "
             "qkv/proj/MLP matrices inside Swin and global-attention blocks. "
             "'matrix' (default) = every genuinely 2-D weight (also ConvNeXt "
             "pointwise layers and the decoder), "
             "attention/MLP weights only, the regime Muon is designed for; "
             "convolutions go to NAdam (weight decay preserved). 'all' = every "
             ">=2-D weight, the previous behaviour, which on this hybrid "
             "encoder routed 48 of 80 tensors -- every conv, depthwise "
             "included -- through Newton-Schulz orthogonalisation.",
    )
    parser.add_argument("--checkpoint-path", default="checkpoints/stage3_segmentation.pt")
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to a checkpoint saved by this script (model/optimizer/"
             "scheduler state) to resume from. Continues at checkpoint['epoch']"
             " + 1 and keeps its best_val_miou, so a later worse epoch here "
             "won't overwrite --checkpoint-path with a regression. Reuse the "
             "same --epochs as the original run -- the cosine schedule is "
             "recomputed from --epochs and only lines up if it matches. "
             "Requires that checkpoint to have been saved with "
             "--save-optimizer-state.",
    )
    parser.add_argument(
        "--save-optimizer-state", action="store_true",
        help="Include optimizer/scheduler state in the saved checkpoint, so "
             "a later --resume-from can continue training exactly where this "
             "run left off. Off by default: optimizer state is comparable in "
             "size to the model weights, and most runs only need the "
             "weights for inference/eval.",
    )
    parser.add_argument(
        "--per-run-checkpoint", action="store_true",
        help="Insert the wandb run id into the checkpoint and history "
             "filenames. Sweep agents all share one command line, so without "
             "this every trial writes its best model to the same path and only "
             "the last one survives.",
    )
    parser.add_argument("--history-path", default="outputs/stage3_segmentation_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_arguments(parser)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    return {
        "stage": "stage3_segmentation_finetune",
        "seed": args.seed,
        "data": {
            "loader_factory": args.loader_factory,
            "loader_config_path": args.loader_config_path,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "num_workers": args.num_workers,
            "num_classes": args.num_classes,
            "ignore_index": args.ignore_index,
            # Context crops for the context-aware encoder; None keeps the
            # loader on its existing (context-free) path.
            "use_context": bool(args.use_context),
            "context_cache_dir": args.context_cache_dir,
            "crop_cache_dir": args.crop_cache_dir,
            "loss_kind": args.loss_kind,
            "lovasz_weight": args.lovasz_weight,
            "focal_gamma": args.focal_gamma,
            "logit_adjust_tau": args.logit_adjust_tau,
            "class_weight_scheme": args.class_weight_scheme,
            "class_weight_clip_max": args.class_weight_clip_max,
            "spatial_jitter_px": args.spatial_jitter_px,
            "brightness_jitter": args.brightness_jitter,
            "contrast_jitter": args.contrast_jitter,
            "train_fraction": args.train_fraction,
            "val_fraction_of_split": args.val_fraction_of_split,
            "ig_loss_weight": args.ig_loss_weight,
            "num_classes_ig": args.num_classes_ig,
            "dc_to_ig_path": args.dc_to_ig_path,
        },
        "runtime": {
            "compile": args.compile,
            "channels_last": args.channels_last,
            "amp_dtype": args.amp_dtype,
        },
        "model": {
            "model_kind": args.model_kind,
            "in_channels": 1,
            "decoder_channels": args.decoder_channels,
            "decoder_dropout": args.decoder_dropout,
            "use_aspp": args.use_aspp,
            "aspp_rates": tuple(args.aspp_rates),
            "window_size": args.window_size,
            "drop_path": args.drop_path,
            # simmim single-branch
            "global_base_grid": args.global_base_grid,
            "hybrid_s3_depth": args.hybrid_s3_depth,
            "hybrid_s4_global": args.hybrid_s4_block == "global",
            # legacy context two-branch (ignored when model_kind == 'simmim')
            "local_base_channels": args.local_base_channels,
            "context_base_channels": args.context_base_channels,
            "context_dim": args.context_dim,
            "swin_depths": tuple(args.swin_depths),
            "swin_num_heads": tuple(args.swin_num_heads),
            "use_stage32": not args.disable_stage32,
            # None -> leave the encoder family default (context on); the
            # variant sweeps set it explicitly.
            **({} if args.use_context is None else {"use_context": args.use_context}),
            "context_fusion": args.context_fusion,
            "uncertainty_head": None if args.uncertainty_head == "none" else {
                "head_type": args.uncertainty_head,
                "num_inducing": args.gp_num_inducing,
                "kernel_scale": args.gp_kernel_scale,
                "num_factors": args.het_num_factors,
                "temperature": args.het_temperature,
                "train_mc_samples": args.het_train_mc_samples,
                "test_mc_samples": args.het_test_mc_samples,
            },
        },
        "initialization": {
            "encoder_checkpoint": args.encoder_checkpoint,
            "strict_checkpoint_load": args.strict_checkpoint_load,
            "freeze_encoder_epochs": (
                0 if args.random_init_encoder else args.freeze_encoder_epochs
            ),
            "random_init_encoder": args.random_init_encoder,
            "variant_id": args.variant_id,
            "variant_metrics_path": args.variant_metrics_path,
        },
        "optimization": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_fraction": args.warmup_fraction,
            "use_muon": args.use_muon,
            "muon_scope": args.muon_scope,
            "muon_lr_mode": args.muon_lr_mode,
            "grad_clip_norm": args.grad_clip_norm,
            "allow_nonfinite_loss": args.allow_nonfinite_loss,
            "llrd": args.llrd,
            "ema_decay": args.ema_decay,
            "muon_lr": args.muon_lr,
            "muon_momentum": args.muon_momentum,
            "muon_weight_decay": args.muon_weight_decay,
            "nadam_lr": args.learning_rate if args.nadam_lr is None else args.nadam_lr,
            "nadam_weight_decay": (args.weight_decay if args.nadam_weight_decay is None
                                   else args.nadam_weight_decay),
            "nadam_beta1": args.nadam_beta1,
            "nadam_beta2": args.nadam_beta2,
        },
        "output": {
            "checkpoint_path": args.checkpoint_path,
            "resume_from": args.resume_from,
            "save_optimizer_state": args.save_optimizer_state,
            "per_run_checkpoint": args.per_run_checkpoint,
            "history_path": args.history_path,
        },
    }


def _load_loader_kwargs(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    try:
        from vision_backend.training.utils import resolve_path
    except ModuleNotFoundError:
        from training.utils import resolve_path
    path = resolve_path(config_path)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise TypeError("Loader config JSON must deserialize to a dictionary.")
    return loaded


def train_stage(config: dict, wandb_run=None) -> dict:
    import torch
    try:
        from vision_backend.model.optimizers import (
            build_routed_muon_nadam_optimizer,
            create_cosine_scheduler_with_warmup,
            create_optimizer,
        )
        from vision_backend.training.builders import (
            build_context_segmentation_model,
            build_simmim_segmentation_model,
            load_encoder_from_pretrainer_checkpoint,
            load_simmim_encoder_checkpoint,
        )
        from vision_backend.training.ema import ModelEMA
        from vision_backend.training.hierarchy import (
            aggregate_dc_class_counts_to_ig,
            build_dc_to_ig_tensor,
            load_dc_to_ig_mapping,
            num_ig_classes,
        )
        from vision_backend.training.utils import (
            compute_class_weights,
            compute_log_class_priors,
            count_parameters,
            freeze_module,
            load_checkpoint,
            load_loader_bundle,
            resolve_path,
            run_segmentation_epoch,
            save_checkpoint,
            save_history,
            select_device,
            set_seed,
            unfreeze_module,
        )
        from vision_backend.seg_dataset import load_or_compute_class_pixel_counts
    except ModuleNotFoundError:
        from model.optimizers import (
            build_routed_muon_nadam_optimizer,
            create_cosine_scheduler_with_warmup,
            create_optimizer,
        )
        from training.builders import (
            build_context_segmentation_model,
            build_simmim_segmentation_model,
            load_encoder_from_pretrainer_checkpoint,
            load_simmim_encoder_checkpoint,
        )
        from training.ema import ModelEMA
        from training.hierarchy import (
            aggregate_dc_class_counts_to_ig,
            build_dc_to_ig_tensor,
            load_dc_to_ig_mapping,
            num_ig_classes,
        )
        from training.utils import (
            compute_class_weights,
            compute_log_class_priors,
            count_parameters,
            freeze_module,
            load_checkpoint,
            load_loader_bundle,
            resolve_path,
            run_segmentation_epoch,
            save_checkpoint,
            save_history,
            select_device,
            set_seed,
            unfreeze_module,
        )
        from seg_dataset import load_or_compute_class_pixel_counts

    set_seed(torch, int(config["seed"]))
    device = select_device(torch)
    use_amp = device.type == "cuda"
    runtime = config.get("runtime", {})
    if device.type == "cuda":
        # Fixed 512x512 crops every step, so cuDNN can pick and reuse the best
        # algorithm instead of re-heuristing per call.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_config = config["data"]
    model_config = dict(config["model"])
    initialization = config["initialization"]
    optimization = config["optimization"]
    output = config["output"]

    loader_kwargs = _load_loader_kwargs(data_config["loader_config_path"])
    loader_kwargs.update(
        {
            "batch_size": int(data_config["batch_size"]),
            "num_workers": int(data_config["num_workers"]),
            "seed": int(config["seed"]),
            "train_fraction": float(data_config.get("train_fraction", 1.0)),
            "val_fraction": float(data_config.get("val_fraction_of_split", 1.0)),
            "spatial_jitter_px": int(data_config.get("spatial_jitter_px", 32)),
            "brightness_jitter": float(data_config.get("brightness_jitter", 0.15)),
            "contrast_jitter": float(data_config.get("contrast_jitter", 0.15)),
        }
    )
    # The padded crop cache is the difference between slicing a memmap and
    # re-decompressing GeoTIFF tiles for every crop of every epoch, so say
    # loudly which path this run is on rather than leaving it to be inferred
    # from the epoch time.
    if data_config.get("crop_cache_dir"):
        loader_kwargs["cache_dir"] = data_config["crop_cache_dir"]
    if loader_kwargs.get("cache_dir"):
        print(f"[stage3] crop cache: {loader_kwargs['cache_dir']}")
    else:
        print(
            "[stage3] WARNING: no crop cache (--crop-cache-dir / loader-config "
            "`cache_dir`) -- every crop is read live from the source GeoTIFFs "
            "and spatial jitter forces an up-to-4-tile DEFLATE decompression "
            "per crop per epoch. Build one with prep_seg_crop_cache.py."
        )
    # Only pass the context knobs when the run actually wants context, so a
    # loader factory without those parameters is unaffected.
    if data_config.get("use_context"):
        loader_kwargs["use_context"] = True
        if data_config.get("context_cache_dir"):
            loader_kwargs["context_cache_dir"] = data_config["context_cache_dir"]
    loaders = load_loader_bundle(data_config["loader_factory"], loader_kwargs)
    if "train" not in loaders or "val" not in loaders:
        raise KeyError("Loader bundle must contain at least `train` and `val` loaders.")

    num_classes = data_config.get("num_classes")
    if num_classes is None:
        num_classes = loaders.get("num_classes")
    if num_classes is None:
        raise ValueError(
            "num_classes was not provided and could not be inferred from the loader bundle."
        )
    model_config["num_classes"] = int(num_classes)

    weight_scheme = data_config.get("class_weight_scheme", "none")
    loss_kind = data_config.get("loss_kind", "ce")
    needs_long_tail_priors = loss_kind in ("balanced_softmax", "logit_adjusted")

    # F1: hierarchical IG aux head setup. Must happen before model
    # construction so decoder.head_ig gets built to the right size; 0.0 (the
    # default) leaves dc_to_ig_tensor/num_classes_ig as None, which builds no
    # head at all (see training/segmentation.py's neutral-default contract).
    ig_loss_weight = float(data_config.get("ig_loss_weight", 0.0))
    dc_to_ig_tensor = None
    num_classes_ig = None
    if ig_loss_weight > 0:
        dc_to_ig_path = data_config.get("dc_to_ig_path")
        dc_to_ig_mapping = load_dc_to_ig_mapping(
            str(resolve_path(dc_to_ig_path)) if dc_to_ig_path else None
        )
        dc_to_ig_tensor = build_dc_to_ig_tensor(torch, int(num_classes), mapping=dc_to_ig_mapping)
        num_classes_ig = int(data_config.get("num_classes_ig") or num_ig_classes(dc_to_ig_mapping))
        model_config["num_classes_ig"] = num_classes_ig
        print(f"[stage3] IG aux head enabled: {num_classes_ig} IG classes, "
              f"ig_loss_weight={ig_loss_weight}")

    class_weights = None
    dc_log_priors = None
    ig_log_priors = None
    if weight_scheme != "none" or needs_long_tail_priors:
        train_dataset = loaders.get("train_dataset")
        if train_dataset is not None and hasattr(train_dataset, "records"):
            print(f"[stage3] Computing class pixel counts from "
                  f"{len(train_dataset.records)} train crops...")
            class_counts = load_or_compute_class_pixel_counts(
                train_dataset.records,
                imagery_path=train_dataset.imagery_path,
                label_path=train_dataset.label_path,
                num_classes=int(num_classes),
                ignore_index=int(data_config["ignore_index"]),
                cache_dir=train_dataset.cache_dir,
                manifest_path=loader_kwargs.get("manifest_path"),
            )
            print(f"[stage3] class pixel counts: {class_counts}")
            if weight_scheme != "none":
                class_weights = compute_class_weights(
                    torch, class_counts, int(num_classes),
                    scheme=weight_scheme,
                    clip_max=float(data_config.get("class_weight_clip_max", 10.0)),
                )
                print(f"[stage3] class weights: {[round(w, 3) for w in class_weights.tolist()]}")
            if needs_long_tail_priors:
                # F2: priors from the same train-split pixel counts, not a
                # second raster pass -- and, when F1 is also on, the IG priors
                # are just the DC counts summed per IG group (no second scan).
                dc_log_priors = compute_log_class_priors(torch, class_counts, int(num_classes))
                if dc_to_ig_tensor is not None:
                    ig_counts = aggregate_dc_class_counts_to_ig(class_counts, dc_to_ig_tensor)
                    ig_log_priors = compute_log_class_priors(torch, ig_counts, num_classes_ig)
        else:
            print(
                "[stage3] WARNING: loader bundle has no train_dataset.records "
                f"(custom --loader-factory?) -- cannot compute class weights/priors, "
                f"training unweighted/unadjusted despite "
                f"--class-weight-scheme={weight_scheme} / --loss-kind={loss_kind}."
            )

    model_kind = model_config.get("model_kind", "simmim")
    strict = bool(initialization["strict_checkpoint_load"])
    random_init = bool(initialization.get("random_init_encoder", False))

    if (
        model_kind == "context"
        and model_config.get("use_context", True)
        and model_config.get("context_fusion", "film") == "xattn"
    ):
        # Spatial fusion places context cells relative to the local crop, so it
        # needs the true ratio of window to crop -- taken from the data actually
        # being served, never assumed. Written into config["model"] too: that is
        # what the checkpoint persists (model_config is a working copy), and a
        # reload must rebuild the same geometry.
        ds = loaders.get("train_dataset")
        ctx_size = getattr(ds, "context_size", None)
        crop = ds.records[0].size if ds is not None and getattr(ds, "records", None) else None
        if not ctx_size or not crop:
            raise ValueError(
                "--context-fusion xattn needs the context window and crop sizes from "
                "the loader (SegmentationCropDataset.context_size / records[].size)."
            )
        ratio = float(ctx_size) / float(crop)
        model_config["context_extent_ratio"] = ratio
        config["model"]["context_extent_ratio"] = ratio
        print(f"[stage3] spatial context fusion: {ctx_size}px window over {crop}px "
              f"crops -> extent ratio {ratio:g}")

    if model_kind == "simmim":
        model = build_simmim_segmentation_model(model_config).to(device)
    elif model_kind == "context":
        model = build_context_segmentation_model(model_config).to(device)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r} (expected simmim|context).")

    if random_init:
        # Nothing pretrained to preserve, so the mechanisms that exist to protect
        # pretrained weights are meaningless here and are forced off in
        # train_stage(); see the llrd / freeze_encoder_epochs overrides.
        print("[stage3] Encoder trained FROM SCRATCH (--random-init-encoder): "
              "no checkpoint loaded.")
    else:
        if not initialization.get("encoder_checkpoint"):
            raise ValueError(
                "--encoder-checkpoint is required unless --random-init-encoder is set."
            )
        encoder_ckpt = resolve_path(initialization["encoder_checkpoint"])
        if model_kind == "simmim":
            load_simmim_encoder_checkpoint(
                torch, model.encoder, encoder_ckpt, prefer_ema=True, strict=strict
            )
        else:
            load_encoder_from_pretrainer_checkpoint(
                torch, model.encoder, encoder_ckpt, strict=strict
            )
    print(f"Model kind: {model_kind}"
          + (f" (context branch {'ON' if getattr(model, 'use_context', True) else 'OFF'})"
             if model_kind == "context" else ""))

    if bool(runtime.get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
        print("[stage3] channels_last memory format enabled.")

    # `base_model` always refers to the real module: torch.compile returns an
    # OptimizedModule whose state_dict() keys are prefixed `_orig_mod.`, which
    # would silently produce checkpoints that no longer load into the plain
    # model. Save/freeze/count through base_model, run through `model`.
    base_model = model
    if bool(runtime.get("compile", False)):
        # Compile after the encoder checkpoint load and after any memory-format
        # change, so the compiled graph reflects the final module.
        # dynamic=False: crops are fixed-size, and torch 2.5.1's inductor cannot
        # build the Swin window-attention reduction once a frame that has seen a
        # second shape is recompiled with symbolic H/W ("Failed to find static
        # RBLOCK"). Other shapes (eval batch, throughput probe) get static recompiles.
        model = torch.compile(model, dynamic=False)
        print("[stage3] torch.compile enabled (first step pays the compile cost).")

    # F5: EMA, built from base_model (not the possibly torch.compile'd `model`)
    # so ModelEMA's named_parameters()-keyed update matches by name. 0.0 (the
    # default) keeps ema=None -- no EMA tracking, current behavior.
    ema_decay = float(optimization.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        ema = ModelEMA(base_model, decay=ema_decay).to(device)
        print(f"[stage3] EMA enabled (decay={ema_decay}); val/miou and the "
              "saved checkpoint use the EMA weights.")

    llrd = float(optimization.get("llrd", 1.0))
    if bool(optimization["use_muon"]):
        # Routed hybrid: Muon on 2-D weight matrices, NAdam on everything else,
        # each with its own LR + momentum (all sweepable).
        optimizer = build_routed_muon_nadam_optimizer(
            model,
            muon_lr=float(optimization["muon_lr"]),
            muon_momentum=float(optimization["muon_momentum"]),
            muon_weight_decay=float(optimization["muon_weight_decay"]),
            muon_scope=str(optimization.get("muon_scope", "matrix")),
            muon_lr_mode=str(optimization.get("muon_lr_mode", "absolute")),
            nadam_lr=float(optimization["nadam_lr"]),
            nadam_weight_decay=float(optimization.get("nadam_weight_decay", 0.0)),
            nadam_betas=(float(optimization["nadam_beta1"]), float(optimization["nadam_beta2"])),
            llrd=llrd,
        )
    else:
        optimizer = create_optimizer(
            model,
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            use_muon=False,
            llrd=llrd,
        )
    accum_steps = max(int(data_config.get("accum_steps", 1)), 1)
    steps_per_epoch = -(-max(len(loaders["train"]), 1) // accum_steps)  # ceil div
    total_steps = int(optimization["epochs"]) * steps_per_epoch
    warmup_steps = int(float(optimization["warmup_fraction"]) * total_steps)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    checkpoint_path = resolve_path(output["checkpoint_path"])
    epochs_count = int(optimization["epochs"])
    checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{epochs_count}ep{checkpoint_path.suffix}"
    )
    history_path = resolve_path(output["history_path"])

    if bool(output.get("per_run_checkpoint", False)):
        run_id = getattr(wandb_run, "id", None) if wandb_run is not None else None
        if run_id:
            checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_{run_id}{checkpoint_path.suffix}"
            )
            history_path = history_path.with_name(
                f"{history_path.stem}_{run_id}{history_path.suffix}"
            )
        else:
            print("[stage3] WARNING: --per-run-checkpoint set but no wandb run id "
                  "is available; falling back to the shared path.")

    print(f"Using device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Segmentation model params: {count_parameters(base_model):,}")
    if model_config.get("uncertainty_head"):
        print(f"[stage3] uncertainty head: {base_model.decoder.head}")
    print(f"Encoder params: {count_parameters(base_model.encoder):,}")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches:   {len(loaders['val'])}")

    history_rows: list[dict[str, float]] = []
    best_val_miou = float("-inf")
    # C4: reported alongside DC mIoU so a variant's IG-level score is
    # comparable too; stays None when the IG aux head is off.
    best_val_miou_ig = None
    # Mean wall-clock per training epoch / step, for the cost half of the
    # variant comparison. Measured, not estimated from FLOPs.
    _epoch_seconds: list[float] = []
    _epoch_steps: list[int] = []
    best_epoch = 0
    start_epoch = 1
    encoder_unfrozen = False

    resume_from = output.get("resume_from")
    if resume_from:
        resume_path = resolve_path(resume_from)
        print(f"[stage3] Resuming from checkpoint: {resume_path}")
        resumed = load_checkpoint(torch, resume_path, map_location=device)
        if "optimizer_state" not in resumed or "scheduler_state" not in resumed:
            raise KeyError(
                f"{resume_path} has no optimizer/scheduler state -- it was "
                "saved without --save-optimizer-state (or predates this "
                "flag) -- cannot resume from it; start a fresh run instead "
                "(optionally via --encoder-checkpoint if you just want its "
                "weights as a starting point)."
            )
        base_model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
        if ema is not None:
            if "ema_state" in resumed:
                ema.load_state_dict(resumed["ema_state"])
            else:
                print("[stage3] WARNING: --ema-decay > 0 but the resumed "
                      "checkpoint has no ema_state -- EMA restarts fresh from "
                      "the resumed model_state instead of continuing.")
        start_epoch = int(resumed["epoch"]) + 1
        best_val_miou = float(resumed["metrics"].get("miou", float("-inf")))
        best_epoch = int(resumed["epoch"])
        print(
            f"[stage3] Resumed at epoch {start_epoch} "
            f"(best_val_miou={best_val_miou:.6f} from epoch {best_epoch})."
        )

    freeze_encoder_epochs = int(initialization["freeze_encoder_epochs"])
    if freeze_encoder_epochs > 0:
        freeze_module(base_model.encoder)

    for epoch in range(start_epoch, int(optimization["epochs"]) + 1):
        _epoch_start = time.time()
        if freeze_encoder_epochs > 0 and epoch > freeze_encoder_epochs and not encoder_unfrozen:
            unfreeze_module(base_model.encoder)
            encoder_unfrozen = True
            print(f"[stage3] Unfroze encoder at epoch {epoch}.")

        train_metrics = run_segmentation_epoch(
            torch,
            model,
            loaders["train"],
            device,
            num_classes=int(num_classes),
            ignore_index=int(data_config["ignore_index"]),
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            amp_dtype=runtime.get("amp_dtype", "fp16"),
            accum_steps=accum_steps,
            class_weights=class_weights,
            loss_kind=loss_kind,
            focal_gamma=float(data_config.get("focal_gamma", 2.0)),
            logit_adjust_tau=float(data_config.get("logit_adjust_tau", 1.0)),
            dc_log_priors=dc_log_priors,
            grad_clip_norm=float(optimization.get("grad_clip_norm", 0.0)) or None,
            error_on_nonfinite_loss=not bool(optimization.get("allow_nonfinite_loss", False)),
            dc_to_ig=dc_to_ig_tensor,
            num_classes_ig=num_classes_ig,
            ig_loss_weight=ig_loss_weight,
            ig_log_priors=ig_log_priors,
            lovasz_weight=float(data_config.get("lovasz_weight", 0.0)),
            ema=ema,
            ema_source_model=base_model,
            progress_desc=f"Stage3 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [train]",
            leave_progress=True,
        )
        # F5: val/miou (and the checkpoint below) come from the EMA weights
        # once EMA is active -- ema is None (default) leaves this as `model`,
        # today's behavior exactly.
        eval_model = ema.ema if ema is not None else model
        val_metrics = run_segmentation_epoch(
            torch,
            eval_model,
            loaders["val"],
            device,
            num_classes=int(num_classes),
            ignore_index=int(data_config["ignore_index"]),
            optimizer=None,
            use_amp=False,
            error_on_nonfinite_loss=not bool(optimization.get("allow_nonfinite_loss", False)),
            dc_to_ig=dc_to_ig_tensor,
            num_classes_ig=num_classes_ig,
            ig_loss_weight=ig_loss_weight,
            progress_desc=f"Stage3 Epoch {epoch:02d}/{int(optimization['epochs']):02d} [val]",
            leave_progress=True,
        )
        if device.type == "cuda":
            # Train (fp16 autocast) and val (fp32, no-grad) leave differently
            # shaped blocks in the caching allocator; on a near-full GPU that
            # fragmentation can OOM the next epoch even with enough nominal
            # free memory. Reset the pool at the epoch boundary.
            torch.cuda.empty_cache()
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_pixel_acc": float(train_metrics["pixel_acc"]),
            "train_miou": float(train_metrics["miou"]),
            "val_loss": float(val_metrics["loss"]),
            "val_pixel_acc": float(val_metrics["pixel_acc"]),
            "val_miou": float(val_metrics["miou"]),
            "lr": current_lr,
        }
        wandb_row = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/pixel_acc": train_metrics["pixel_acc"],
            "train/miou": train_metrics["miou"],
            "val/loss": val_metrics["loss"],
            "val/pixel_acc": val_metrics["pixel_acc"],
            "val/miou": val_metrics["miou"],
            "optimizer/lr": current_lr,
        }
        if "miou_ig" in val_metrics:
            # F1: only present (in both history and wandb) when the IG aux
            # head is active -- neutral default keeps today's plain columns.
            row["train_miou_ig"] = float(train_metrics["miou_ig"])
            row["val_miou_ig"] = float(val_metrics["miou_ig"])
            wandb_row["train/miou_ig"] = train_metrics["miou_ig"]
            wandb_row["val/miou_ig"] = val_metrics["miou_ig"]
        history_rows.append(row)
        log_metrics(wandb_run, wandb_row, step=epoch)

        ig_suffix = f" val_miou_ig={val_metrics['miou_ig']:.6f}" if "miou_ig" in val_metrics else ""
        print(
            f"[stage3 epoch {epoch:02d}/{int(optimization['epochs']):02d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_miou={val_metrics['miou']:.6f}{ig_suffix} "
            f"lr={current_lr:.3e}"
        )

        _epoch_seconds.append(time.time() - _epoch_start)
        _epoch_steps.append(max(1, len(loaders["train"])))
        if "ig_miou" in val_metrics:
            ig_now = float(val_metrics["ig_miou"])
            best_val_miou_ig = (
                ig_now if best_val_miou_ig is None else max(best_val_miou_ig, ig_now)
            )

        if float(val_metrics["miou"]) > best_val_miou:
            best_val_miou = float(val_metrics["miou"])
            best_epoch = epoch
            # F5: the EMA weights ARE the primary checkpoint once EMA is
            # active (ema is None by default -> base_model.state_dict(),
            # today's behavior exactly).
            checkpoint_state = {
                "stage": config["stage"],
                "epoch": epoch,
                "model_state": ema.ema.state_dict() if ema is not None else base_model.state_dict(),
                "metrics": dict(val_metrics),
                "config": config,
            }
            if ema is not None:
                checkpoint_state["ema_state"] = ema.state_dict()
            if bool(output.get("save_optimizer_state", False)):
                checkpoint_state["optimizer_state"] = optimizer.state_dict()
                checkpoint_state["scheduler_state"] = scheduler.state_dict()
            save_checkpoint(torch, checkpoint_state, checkpoint_path)

    # Epoch wall-clock includes validation; dividing by train steps gives a
    # consistent per-step figure across variants, which is what is compared.
    mean_train_step_ms = (
        (sum(_epoch_seconds) / sum(_epoch_steps)) * 1000.0
        if _epoch_seconds and sum(_epoch_steps)
        else None
    )

    # --- C4: per-run measurements for the variant comparison ----------------
    try:
        from vision_backend.training.variant_metrics import (
            measure_inference_throughput,
            parameter_breakdown,
            peak_memory_gb,
            write_variant_row,
        )
    except ModuleNotFoundError:
        from training.variant_metrics import (
            measure_inference_throughput,
            parameter_breakdown,
            peak_memory_gb,
            write_variant_row,
        )

    variant_row = {
        # build_config puts these in `initialization`, not `data` -- reading
        # them from data_config silently disabled the whole variant comparison:
        # every row was written with variant_id=None, and the metrics file was
        # never written at all because the guard below always saw None.
        "variant_id": initialization.get("variant_id"),
        "run_name": getattr(wandb_run, "name", None),
        "run_id": getattr(wandb_run, "id", None),
        "model_kind": model_kind,
        "use_context": bool(model_config.get("use_context", True)),
        "local_base_channels": model_config.get("local_base_channels"),
        "decoder_channels": model_config.get("decoder_channels"),
        "epochs": int(optimization["epochs"]),
        "best_val_miou": best_val_miou,
        "best_val_miou_ig": best_val_miou_ig,
        "best_epoch": best_epoch,
        "train_step_ms": mean_train_step_ms,
        "peak_mem_gb": peak_memory_gb(torch, device),
        **parameter_breakdown(model),
    }
    try:
        variant_row.update(
            measure_inference_throughput(
                torch, model, device,
                input_size=int(data_config.get("crop_size", 512)),
                batch_size=2,
                needs_context=bool(model_config.get("use_context", True)),
            )
        )
    except Exception as exc:  # never fail a finished run over a benchmark
        print(f"[stage3] inference throughput measurement skipped: {exc}")

    log_metrics(wandb_run, {f"variant/{k}": v for k, v in variant_row.items()
                            if isinstance(v, (int, float))})
    if initialization.get("variant_metrics_path"):
        write_variant_row(initialization["variant_metrics_path"], variant_row)
        print(f"[stage3] variant row -> {initialization['variant_metrics_path']}")

    save_history(history_rows, history_path)
    final_metrics = {
        "best_val_miou": best_val_miou,
        "best_epoch": float(best_epoch),
        "total_params": float(count_parameters(base_model)),
    }
    log_metrics(wandb_run, final_metrics)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved history to {history_path}")
    return final_metrics


def main() -> int:
    args = parse_args()
    base_config = build_config(args)

    if maybe_run_sweep(
        args,
        stage_name="stage3_segmentation_finetune",
        base_config=base_config,
        train_fn=train_stage,
    ):
        return 0

    run = init_wandb_run(args, base_config, stage_name="stage3_segmentation_finetune")
    try:
        merged_config = merge_wandb_config(base_config, run)
        train_stage(merged_config, run)
    finally:
        finish_wandb_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
