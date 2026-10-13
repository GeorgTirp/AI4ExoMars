#!/usr/bin/env python3
"""Stage 4: fine-tune a Stage-3 segmenter into a block-level classifier.

One class per 512x512 block (the planetary scientists' labels): the 14 NOAH-H
DC classes plus new ones. Model: model/block_head.py -- the segmenter's
dominant-class prior + an attention-MIL head, so epoch 0 is the zero-shot
segmenter on the NOAH-H classes. Data: block_dataset.py manifest.

Schedule (LP-FT, Kumar et al., "Fine-Tuning can Distort Pretrained Features
and Underperform Out-of-Distribution", ICLR 2022): `--probe-epochs` with only
the new head training on the frozen segmenter, then everything. The head uses
AdamW at --head-lr throughout; the segmenter gets the Stage-3 optimizer (Muon
on transformer matrices with match_adam LR transfer + NAdamW elsewhere) at
fine-tuning LRs, created only when it unfreezes -- Muon zero-fills missing
gradients and still applies weight decay, so a Muon optimizer over frozen
weights would shrink them. Both optimizers warm up and decay with cosine over
their own active steps.

Model selection: val macro-F1 over the classes with >= --select-min-support val
blocks (class-imbalanced data: accuracy would reward the common classes; a
class with 1-2 val blocks would make the choice hinge on single blocks).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from vision_backend.block_dataset import (DEFAULT_DC_CLASSES_JSON, BlockClasses, BlockDataset,
                                              SizeGroupedBatchSampler, load_block_manifest)
    from vision_backend.model.block_head import BlockClassifier
    from vision_backend.model.optimizers import build_routed_muon_nadam_optimizer, create_cosine_scheduler_with_warmup
    from vision_backend.training.builders import build_simmim_segmentation_model, load_segmentation_model_from_checkpoint
    from vision_backend.training.utils import compute_class_weights, count_parameters, set_seed
    from vision_backend.training.wandb_utils import add_wandb_arguments, finish_wandb_run, init_wandb_run, log_metrics
    from vision_backend.laplace_prior import DiagonalLaplacePrior
except ModuleNotFoundError:
    from block_dataset import (DEFAULT_DC_CLASSES_JSON, BlockClasses, BlockDataset,
                               SizeGroupedBatchSampler, load_block_manifest)
    from model.block_head import BlockClassifier
    from model.optimizers import build_routed_muon_nadam_optimizer, create_cosine_scheduler_with_warmup
    from training.builders import build_simmim_segmentation_model, load_segmentation_model_from_checkpoint
    from training.utils import compute_class_weights, count_parameters, set_seed
    from training.wandb_utils import add_wandb_arguments, finish_wandb_run, init_wandb_run, log_metrics
    from laplace_prior import DiagonalLaplacePrior

DEFAULT_IMAGERY = "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/drg_on_label_grid.tif"
# HybridEncoder at the tuned recipe, seed 42 (E10 "no_lovasz", val mIoU 0.2072).
DEFAULT_SEG_CHECKPOINT = "checkpoints/arch_ablation_no_lovasz/best_16ep.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4: block-level fine-tuning of the Stage-3 segmenter.")
    p.add_argument("--seg-checkpoint", default=DEFAULT_SEG_CHECKPOINT)
    p.add_argument("--manifest", required=True, help="Block manifest CSV (see block_dataset.py).")
    p.add_argument("--imagery-path", default=DEFAULT_IMAGERY, help="Mosaic for col,row blocks.")
    p.add_argument("--dc-classes-json", default=DEFAULT_DC_CLASSES_JSON)
    p.add_argument("--block-classes", default=None,
                   help="Fixed class list (block_classes.json from prep_expert_block_manifest.py); "
                        "default: NOAH-H classes + the new classes found in the manifest.")
    p.add_argument("--expect-new-classes", type=int, default=None,
                   help="Fail unless the labels contain exactly this many non-NOAH-H classes "
                        "(catches a misspelt NOAH-H name turning into a spurious new class).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--probe-epochs", type=int, default=2,
                   help="Head-only epochs on the frozen segmenter before full fine-tuning (LP-FT).")
    p.add_argument("--freeze-segmenter", action="store_true", help="Never unfreeze: a pure probe.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--head-weight-decay", type=float, default=1e-4)
    # Segmenter: 0.1x the tuned Stage-3 values (E7 trial f1qeno3u) -- fine-tuning, not training.
    p.add_argument("--muon-lr", type=float, default=1.72e-5, help="AdamW-equivalent (match_adam).")
    p.add_argument("--muon-weight-decay", type=float, default=0.0109)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--nadam-lr", type=float, default=6.08e-6)
    p.add_argument("--nadam-weight-decay", type=float, default=0.0831)
    p.add_argument("--llrd", type=float, default=1.0)
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--class-weight-scheme", default="inverse_sqrt",
                   choices=("none", "inverse_sqrt", "inverse", "effective_number"))
    p.add_argument("--class-weight-clip-max", type=float, default=10.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--attn-hidden", type=int, default=128)
    p.add_argument("--jitter", type=int, default=32, help="Training shift (px) for mosaic blocks.")
    p.add_argument("--block-size", type=int, default=512,
                   help="Backbone input size; larger blocks (multiples) are tiled. --batch-size counts blocks "
                        "of this size (a 1024 block counts 4).")
    p.add_argument("--amp-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true", help="Zero-shot evaluation of the initial model only.")
    p.add_argument("--laplace-prior", default=None,
                   help="Campaign-0 prior (scripts/fit_laplace_prior.py); off unless --laplace-tau > 0.")
    p.add_argument("--laplace-tau", type=float, default=0.0,
                   help="Prior strength: effective number of NOAH-H images (precision = tau * F_img).")
    p.add_argument("--laplace-scope", choices=("all", "encoder"), default="all",
                   help="encoder: no prior on the decoder / segmentation head.")
    p.add_argument("--select-min-support", type=int, default=5,
                   help="Model selection: val macro-F1 over classes with at least this many val blocks.")
    add_wandb_arguments(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def summarize(logits: np.ndarray, labels: np.ndarray, classes: BlockClasses, *, n_bins: int = 15,
              min_support: int = 5) -> dict:
    """Block metrics. Macro averages run over the classes present in `labels`;
    `macro_f1_sel` (model selection) only over classes with >= min_support
    blocks, so a class with one or two val blocks cannot swing the choice."""
    K = len(classes.names)
    logits = logits.astype(np.float64)
    logp = logits - logits.max(1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    prob = np.exp(logp)
    pred = prob.argmax(1)
    conf = np.zeros((K, K), dtype=np.int64)
    np.add.at(conf, (labels, pred), 1)
    tp = np.diag(conf).astype(np.float64)
    support, predicted = conf.sum(1), conf.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / support
        precision = tp / predicted
        f1 = 2 * tp / (support + predicted)
    present = support > 0
    is_new = np.array([s is None for s in classes.seg_index])

    top = prob.max(1)
    correct = pred == labels
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top > lo) & (top <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - top[m].mean())

    def macro(values, mask):
        return float(np.mean(values[mask])) if mask.any() else float("nan")

    return {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": macro(recall, present),
        "macro_f1": macro(f1, present),
        "macro_f1_sel": macro(f1, support >= max(1, min_support)),
        "balanced_accuracy_noahh": macro(recall, present & ~is_new),
        "balanced_accuracy_new": macro(recall, present & is_new),
        "macro_f1_noahh": macro(f1, present & ~is_new),
        "macro_f1_new": macro(f1, present & is_new),
        "nll": float(-logp[np.arange(len(labels)), labels].mean()),
        "ece": float(ece),
        "per_class": [
            {"name": classes.names[k], "new": bool(is_new[k]), "support": int(support[k]),
             "recall": None if not present[k] else float(recall[k]),
             "precision": None if predicted[k] == 0 else float(precision[k]),
             "f1": None if not present[k] else float(f1[k])}
            for k in range(K)
        ],
        "confusion": conf.tolist(),
    }


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(image, valid)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def seg_model_config_from(seg_model, seg_config: dict) -> dict:
    """The Stage-3 model config with the head sizes filled in, enough to
    rebuild the segmenter without its original checkpoint."""
    cfg = dict(seg_config.get("model", {}) or {})
    cfg["num_classes"] = int(seg_model.decoder.head.out_channels)
    if seg_model.decoder.head_ig is not None:
        cfg["num_classes_ig"] = int(seg_model.decoder.head_ig.out_channels)
    cfg.setdefault("in_channels", 1)
    return cfg


def load_block_classifier(path: str | Path, device: str = "cpu") -> tuple[BlockClassifier, dict]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    seg_model = build_simmim_segmentation_model(ckpt["seg_model_config"])
    head = ckpt["head_config"]
    model = BlockClassifier(seg_model, ckpt["classes"]["seg_index"],
                            attn_hidden=head["attn_hidden"], dropout=head["dropout"],
                            tile_size=head.get("tile_size", 512))
    model.load_state_dict(ckpt["block_state"])
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------------------
def make_loader(dataset, *, batch_size, shuffle, num_workers, base):
    sampler = SizeGroupedBatchSampler(dataset.block_sides(), batch_size, base=base, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def set_segmenter_trainable(model: BlockClassifier, trainable: bool) -> None:
    for p in model.seg.parameters():
        p.requires_grad_(trainable)


def head_optimizer(model: BlockClassifier, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    named = [(n, p) for n, p in model.named_parameters() if not n.startswith("seg.")]
    decay = [p for n, p in named if p.ndim >= 2]
    no_decay = [p for n, p in named if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def main() -> None:
    args = parse_args()
    set_seed(torch, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and args.amp_dtype == "bf16"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, classes = load_block_manifest(args.manifest, args.dc_classes_json, args.block_classes)
    if args.expect_new_classes is not None and classes.num_new != args.expect_new_classes:
        new = [n for n, s in zip(classes.names, classes.seg_index) if s is None]
        raise SystemExit(f"Expected {args.expect_new_classes} new classes, the labels give "
                         f"{classes.num_new}: {new} -- check the NOAH-H class spellings.")
    splits = {s: [r for r in records if r.split == s] for s in ("train", "val", "test")}
    if not splits["val"] or (not splits["train"] and not args.eval_only):
        raise SystemExit(f"Manifest needs train and val blocks; got { {s: len(v) for s, v in splits.items()} }")

    K = len(classes.names)
    counts = {s: np.bincount([r.label for r in recs], minlength=K) for s, recs in splits.items()}
    print(f"[stage4] {K} classes ({classes.num_new} new); blocks: "
          + ", ".join(f"{s} {len(v)}" for s, v in splits.items()))
    for k, name in enumerate(classes.names):
        tag = "NEW" if classes.seg_index[k] is None else f"DC{classes.seg_index[k] + 1:>2}"
        print(f"  {k:>2} {tag:<4} {name:<52} train {counts['train'][k]:>5}  val {counts['val'][k]:>4}  "
              f"test {counts['test'][k]:>4}")

    seg_model, model_kind, num_seg, seg_config = load_segmentation_model_from_checkpoint(
        Path(args.seg_checkpoint), device="cpu")
    if model_kind != "simmim":
        raise SystemExit(f"{args.seg_checkpoint}: model_kind {model_kind!r}, the block head needs 'simmim'")
    if num_seg != K - classes.num_new:
        raise SystemExit(f"segmenter has {num_seg} classes, the class list has {K - classes.num_new} NOAH-H classes")
    model = BlockClassifier(seg_model, classes.seg_index, attn_hidden=args.attn_hidden,
                            dropout=args.head_dropout, tile_size=args.block_size).to(device)
    print(f"[stage4] segmenter {args.seg_checkpoint}: {count_parameters(model.seg):,} params; "
          f"block head {sum(p.numel() for p in model.head_parameters()):,} params")

    t0 = time.time()
    datasets = {s: BlockDataset(recs, imagery_path=args.imagery_path, augment=(s == "train"),
                                block_size=args.block_size,
                                jitter=args.jitter if s == "train" else 0)
                for s, recs in splits.items() if recs}
    print(f"[stage4] blocks read in {time.time() - t0:.0f}s")
    loaders = {s: make_loader(d, batch_size=args.batch_size, shuffle=(s == "train"),
                              num_workers=args.num_workers, base=args.block_size)
               for s, d in datasets.items()}

    config = {"args": vars(args), "classes": {"names": classes.names, "seg_index": classes.seg_index},
              "counts": {s: c.tolist() for s, c in counts.items()}}
    wandb_run = init_wandb_run(args, config, stage_name="stage4_block")

    def evaluate(split: str) -> dict:
        logits, labels = predict(model, loaders[split], device, amp)
        return summarize(logits, labels, classes, min_support=args.select_min_support)

    def scalars(prefix: str, m: dict) -> dict:
        return {f"{prefix}/{k}": v for k, v in m.items() if isinstance(v, (int, float))}

    zero_shot = evaluate("val")
    print(f"[stage4] epoch 0 (zero-shot segmenter prior): val acc {zero_shot['accuracy']:.4f}  "
          f"bal-acc NOAH-H {zero_shot['balanced_accuracy_noahh']:.4f}  macro-F1 {zero_shot['macro_f1']:.4f}")
    log_metrics(wandb_run, {"epoch": 0, **scalars("val", zero_shot)}, step=0)
    results = {"zero_shot_val": zero_shot}
    if args.eval_only:
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
        finish_wandb_run(wandb_run)
        return

    weights = None
    if args.class_weight_scheme != "none":
        weights = compute_class_weights(torch, counts["train"].tolist(), K, scheme=args.class_weight_scheme,
                                        clip_max=args.class_weight_clip_max).float().to(device)
        print(f"[stage4] class weights: {[round(w, 3) for w in weights.tolist()]}")

    steps_per_epoch = len(loaders["train"])
    probe_epochs = args.epochs if args.freeze_segmenter else min(args.probe_epochs, args.epochs)
    opt_head = head_optimizer(model, args.head_lr, args.head_weight_decay)
    total = args.epochs * steps_per_epoch
    sched_head = create_cosine_scheduler_with_warmup(opt_head, int(args.warmup_fraction * total), total)
    opt_seg = sched_seg = laplace = None
    set_segmenter_trainable(model, False)
    n_train = len(datasets["train"])

    ckpt_path = out_dir / "best.pt"
    history, best_f1, best_epoch = [], -math.inf, 0
    for epoch in range(1, args.epochs + 1):
        if epoch == probe_epochs + 1:
            set_segmenter_trainable(model, True)
            opt_seg = build_routed_muon_nadam_optimizer(
                model.seg, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum,
                muon_weight_decay=args.muon_weight_decay, nadam_lr=args.nadam_lr,
                nadam_weight_decay=args.nadam_weight_decay, muon_scope="transformer",
                muon_lr_mode="match_adam", llrd=args.llrd)
            remaining = (args.epochs - probe_epochs) * steps_per_epoch
            sched_seg = create_cosine_scheduler_with_warmup(opt_seg, int(args.warmup_fraction * remaining), remaining)
            print(f"[stage4] epoch {epoch}: segmenter unfrozen")
            if args.laplace_prior and args.laplace_tau > 0:
                # The objective is sum_n CE_n + 1/2 (theta-mu)^T Lambda (theta-mu); the loss
                # below is the mean CE, so the prior enters divided by N_train. Weight decay
                # stays as in the plain run (at these LRs it shrinks weights by < 0.3 % over
                # the whole run), so the prior is the only difference between the two arms.
                laplace = DiagonalLaplacePrior(args.laplace_prior, model.seg, tau=args.laplace_tau,
                                               scope=args.laplace_scope, device=device)
                print(f"[stage4] Laplace prior: tau {args.laplace_tau:g}, scope {args.laplace_scope}, "
                      f"{laplace.num_params:,} params, / N_train = {n_train}")

        model.train()
        if opt_seg is None:
            model.seg.eval()  # probe: deterministic features (no drop-path / dropout in the segmenter)
        t0, loss_sum, n_seen, pen_sum, n_steps = time.time(), 0.0, 0, 0.0, 0
        for batch in loaders["train"]:
            image = batch["image"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(image, valid)
            loss = F.cross_entropy(logits.float(), label, weight=weights, label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt_head.zero_grad(set_to_none=True)
            if opt_seg is not None:
                opt_seg.zero_grad(set_to_none=True)
            total_loss = loss
            if laplace is not None:
                pen = laplace.penalty() / n_train
                total_loss = loss + pen
                pen_sum += float(pen)
            total_loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               args.grad_clip_norm)
            opt_head.step()
            sched_head.step()
            if opt_seg is not None:
                opt_seg.step()
                sched_seg.step()
            loss_sum += float(loss) * len(label)
            n_seen += len(label)
            n_steps += 1

        val = evaluate("val")
        row = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "seconds": time.time() - t0,
               "head_lr": opt_head.param_groups[0]["lr"],
               "seg_lr": opt_seg.param_groups[-1]["lr"] if opt_seg is not None else 0.0,
               "prior_scale": float(model.prior_scale), "laplace_penalty": pen_sum / max(n_steps, 1),
               **{k: v for k, v in val.items() if isinstance(v, (int, float))}}
        history.append(row)
        print(f"[stage4] epoch {epoch:>3}: loss {row['train_loss']:.4f}  val acc {val['accuracy']:.4f}  "
              f"bal-acc {val['balanced_accuracy']:.4f} (NOAH-H {val['balanced_accuracy_noahh']:.4f}, "
              f"new {val['balanced_accuracy_new']:.4f})  macro-F1 {val['macro_f1']:.4f}  "
              f"(sel {val['macro_f1_sel']:.4f})  NLL {val['nll']:.3f}  {row['seconds']:.0f}s")
        log_metrics(wandb_run, {"epoch": epoch, "train/loss": row["train_loss"], "lr/head": row["head_lr"],
                                "lr/seg": row["seg_lr"], "prior_scale": row["prior_scale"],
                                "train/laplace_penalty": row["laplace_penalty"],
                                **scalars("val", val)}, step=epoch)
        # NaN when no class reaches --select-min-support val blocks: fall back to
        # plain macro-F1, and always keep at least the first epoch's checkpoint.
        score = val["macro_f1_sel"] if math.isfinite(val["macro_f1_sel"]) else val["macro_f1"]
        score = score if math.isfinite(score) else -math.inf
        if best_epoch == 0 or score > best_f1:
            best_f1, best_epoch = score, epoch
            torch.save({"block_state": model.state_dict(),
                        "seg_model_config": seg_model_config_from(model.seg, seg_config),
                        "classes": {"names": classes.names, "seg_index": classes.seg_index},
                        "head_config": {"attn_hidden": args.attn_hidden, "dropout": args.head_dropout,
                                        "tile_size": args.block_size},
                        "source_checkpoint": str(args.seg_checkpoint),
                        "epoch": epoch, "metrics": val, "args": vars(args)}, ckpt_path)
            results["best_val"] = val
        with (out_dir / "history.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    results["best_epoch"] = best_epoch
    if "test" in loaders:
        model, _ = load_block_classifier(ckpt_path, device=str(device))
        results["test"] = evaluate("test")
        print(f"[stage4] test (best epoch {best_epoch}): acc {results['test']['accuracy']:.4f}  "
              f"bal-acc {results['test']['balanced_accuracy']:.4f}  macro-F1 {results['test']['macro_f1']:.4f}")
        log_metrics(wandb_run, scalars("test", results["test"]), step=args.epochs)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[stage4] best val macro-F1 (classes with >= {args.select_min_support} val blocks) {best_f1:.4f} "
          f"at epoch {best_epoch}; wrote {out_dir}/metrics.json")
    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
