#!/usr/bin/env python3
"""Recompute a proper *global* mIoU for stage-3 segmentation checkpoints.

`run_segmentation_epoch` (training/utils.py) reports `val_miou` as a sample-
size-weighted average of *per-batch* mIoU values -- each batch's mIoU is
itself only averaged over the classes present in that specific batch. With
batch_size=2 and 14 imbalanced classes, that's a non-standard metric, not
directly comparable to how mIoU is normally reported (Cityscapes/VOC/ADE20K,
and presumably the original NOAH-H paper): one confusion matrix accumulated
over pixels from the *entire* validation set, with per-class IoU (and their
mean) computed once at the end from that global matrix.

This script does the standard version: run each checkpoint over every val
crop, accumulate per-class intersection/union pixel counts globally, then
report per-class IoU + mIoU from those totals. No retraining involved.

Pass --baseline to also print, per checkpoint, a per-class IoU delta against
a reference checkpoint (e.g. the plain-CE run) -- the quickest way to see
whether a change (focal loss, class weighting, ...) actually moved the
classes that were stuck at ~0 IoU, rather than just watching the scalar
mIoU move.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NAMES = [
    "Smooth + Featureless",
    "Smooth + Lineated",
    "Textured non-bedrock",
    "Smooth bedrock",
    "Textured bedrock",
    "Rugged bedrock",
    "Fractured bedrock",
    "Continuous + Simple form large ripples",
    "Isolated + Simple form large ripples",
    "Rectilinear form large ripples",
    "Continuous small ripples",
    "Bedrock substrate + Non-continuous small ripples",
    "Non-bedrock substrate + Non-continuous small ripples",
    "Boulder fields",
]

DEFAULT_LOADER_CONFIG = (
    "data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived/seg_loader_DC_aoi.json"
)

# Below this, a class is treated as "collapsed" (model isn't predicting it at
# all) for the purposes of the --baseline delta table's callouts.
COLLAPSED_IOU_THRESHOLD = 0.01


def class_name(idx: int) -> str:
    return CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"


def evaluate_checkpoint(
    ckpt_path: Path,
    *,
    dataloader,
    ignore_index: int,
    device,
    num_classes_from_config,
    torch,
    load_segmentation_model_from_checkpoint,
    parse_segmentation_batch,
    resolve_path,
) -> dict:
    model, model_kind, num_classes, config = load_segmentation_model_from_checkpoint(
        ckpt_path, device=str(device)
    )

    reported_val_miou = None
    try:
        history_path = resolve_path(config["output"]["history_path"])
        if history_path.exists():
            import csv

            rows = list(csv.DictReader(history_path.open()))
            if rows:
                best = max(rows, key=lambda r: float(r["val_miou"]))
                reported_val_miou = float(best["val_miou"])
    except Exception:
        pass

    if num_classes_from_config is not None and num_classes != num_classes_from_config:
        raise ValueError(
            f"{ckpt_path}: checkpoint num_classes={num_classes} != loader "
            f"num_classes={num_classes_from_config}"
        )

    intersection = torch.zeros(num_classes, dtype=torch.int64)
    union = torch.zeros(num_classes, dtype=torch.int64)
    correct_total = 0
    valid_total = 0

    with torch.no_grad():
        for batch in dataloader:
            local, target, context = parse_segmentation_batch(batch)
            local = local.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()
            context_tensor = context.to(device, non_blocking=True).float() if context is not None else None

            logits = model(local, context_tensor) if context_tensor is not None else model(local)
            preds = logits.argmax(dim=1)
            valid_mask = target != ignore_index

            correct_total += int((preds[valid_mask] == target[valid_mask]).sum().item())
            valid_total += int(valid_mask.sum().item())

            for class_index in range(num_classes):
                pred_mask = (preds == class_index) & valid_mask
                target_mask = (target == class_index) & valid_mask
                intersection[class_index] += (pred_mask & target_mask).sum().item()
                union[class_index] += (pred_mask | target_mask).sum().item()

    present = union > 0
    per_class_iou = torch.where(
        present, intersection.double() / union.clamp(min=1).double(), torch.zeros(num_classes, dtype=torch.float64)
    )
    global_miou = float(per_class_iou[present].mean()) if present.any() else 0.0
    global_pixel_acc = correct_total / max(valid_total, 1)

    return {
        "ckpt_path": ckpt_path,
        "num_classes": num_classes,
        "reported_val_miou": reported_val_miou,
        "global_miou": global_miou,
        "global_pixel_acc": global_pixel_acc,
        "present": present,
        "per_class_iou": per_class_iou,
        "intersection": intersection,
        "union": union,
    }


def print_checkpoint_report(result: dict, *, split: str) -> None:
    ckpt_path = result["ckpt_path"]
    num_classes = result["num_classes"]
    present = result["present"]
    per_class_iou = result["per_class_iou"]

    print(f"=== {ckpt_path.name} ===")
    if result["reported_val_miou"] is not None:
        print(f"  training-reported val_miou (per-batch average, best epoch): {result['reported_val_miou']:.4f}")
    print(
        f"  GLOBAL confusion-matrix mIoU ({split}, {int(present.sum())}/{num_classes} classes present): "
        f"{result['global_miou']:.4f}"
    )
    print(f"  GLOBAL pixel accuracy: {result['global_pixel_acc']:.4f}")
    print("  Per-class IoU (present classes only, sorted ascending):")
    rows = [
        (class_name(c), float(per_class_iou[c]), int(result["intersection"][c]), int(result["union"][c]))
        for c in range(num_classes)
        if present[c]
    ]
    rows.sort(key=lambda r: r[1])
    for name, iou, inter, uni in rows:
        print(f"    {name:<58s} IoU={iou:.4f}  (intersection={inter}, union={uni})")
    absent = [class_name(c) for c in range(num_classes) if not present[c]]
    if absent:
        print(f"  Absent from {split} split entirely (no predicted or true pixels): {', '.join(absent)}")
    print()


def print_delta_table(baseline: dict, other: dict) -> None:
    num_classes = other["num_classes"]
    base_present = baseline["present"]
    other_present = other["present"]
    base_iou = baseline["per_class_iou"]
    other_iou = other["per_class_iou"]

    print(f"  --- {other['ckpt_path'].name} vs baseline {baseline['ckpt_path'].name} ---")
    print(f"  global mIoU: {baseline['global_miou']:.4f} -> {other['global_miou']:.4f} "
          f"({other['global_miou'] - baseline['global_miou']:+.4f})")
    print(f"  pixel acc:   {baseline['global_pixel_acc']:.4f} -> {other['global_pixel_acc']:.4f} "
          f"({other['global_pixel_acc'] - baseline['global_pixel_acc']:+.4f})")

    rows = []
    for c in range(num_classes):
        if not base_present[c] and not other_present[c]:
            continue
        b = float(base_iou[c]) if base_present[c] else None
        o = float(other_iou[c]) if other_present[c] else None
        delta = (o - b) if (b is not None and o is not None) else None
        rows.append((class_name(c), b, o, delta))
    # Biggest movers first (None deltas -- appeared/disappeared -- sort last).
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))

    print("  Per-class IoU delta (sorted by biggest gain first):")
    for name, b, o, delta in rows:
        b_str = f"{b:.4f}" if b is not None else "absent"
        o_str = f"{o:.4f}" if o is not None else "absent"
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        flag = ""
        if b is not None and b < COLLAPSED_IOU_THRESHOLD:
            if o is not None and o >= COLLAPSED_IOU_THRESHOLD:
                flag = "  <- moved off zero"
            else:
                flag = "  (still collapsed)"
        print(f"    {name:<58s} {b_str:>8s} -> {o_str:>8s}  ({delta_str}){flag}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="One or more stage-3 checkpoint paths.")
    parser.add_argument(
        "--baseline", default=None,
        help="Reference checkpoint. If given, also prints a per-class IoU delta "
             "table for each of `checkpoints` against this one (evaluated too, "
             "even if not repeated in `checkpoints`).",
    )
    parser.add_argument("--loader-config-path", default=DEFAULT_LOADER_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    args = parser.parse_args()

    import torch

    try:
        from vision_backend.training.builders import load_segmentation_model_from_checkpoint
        from vision_backend.training.utils import (
            load_loader_bundle,
            parse_segmentation_batch,
            resolve_path,
            select_device,
        )
    except ModuleNotFoundError:
        from training.builders import load_segmentation_model_from_checkpoint
        from training.utils import (
            load_loader_bundle,
            parse_segmentation_batch,
            resolve_path,
            select_device,
        )

    device = select_device(torch)
    print(f"Device: {device}")

    loader_config_path = resolve_path(args.loader_config_path)
    loader_kwargs = json.loads(loader_config_path.read_text())
    loader_kwargs.update(
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "augment": False,
        }
    )
    num_classes_from_config = loader_kwargs.get("num_classes")

    loaders = load_loader_bundle("vision_backend.seg_dataset:create_segmentation_dataloaders", loader_kwargs)
    dataloader = loaders[args.split]
    dataset = loaders.get(f"{args.split}_dataset")
    n_crops = len(dataset.records) if dataset is not None and hasattr(dataset, "records") else "?"
    print(f"Split: {args.split} ({n_crops} crops)\n")

    def _evaluate(ckpt_path_str: str) -> dict:
        return evaluate_checkpoint(
            Path(ckpt_path_str),
            dataloader=dataloader,
            ignore_index=args.ignore_index,
            device=device,
            num_classes_from_config=num_classes_from_config,
            torch=torch,
            load_segmentation_model_from_checkpoint=load_segmentation_model_from_checkpoint,
            parse_segmentation_batch=parse_segmentation_batch,
            resolve_path=resolve_path,
        )

    results_by_path: dict[str, dict] = {}

    baseline_result = None
    if args.baseline is not None:
        baseline_result = _evaluate(args.baseline)
        results_by_path[args.baseline] = baseline_result
        print_checkpoint_report(baseline_result, split=args.split)

    for ckpt_path_str in args.checkpoints:
        if ckpt_path_str == args.baseline:
            continue  # already evaluated + printed above
        result = _evaluate(ckpt_path_str)
        results_by_path[ckpt_path_str] = result
        print_checkpoint_report(result, split=args.split)
        if baseline_result is not None:
            print_delta_table(baseline_result, result)


if __name__ == "__main__":
    main()
