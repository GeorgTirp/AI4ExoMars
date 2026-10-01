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


def trained_crop_size(config: dict) -> int | None:
    """Crop size the checkpoint was trained on, read from its training manifest.

    The checkpoint config records the loader config it trained with, not the
    crop size, so follow loader config -> manifest_path -> first row's `size`.
    None when any link is missing (the checkpoint is then evaluated untiled).
    """
    import csv

    try:
        loader_cfg = json.loads(Path(config["data"]["loader_config_path"]).read_text())
        with open(loader_cfg["manifest_path"], newline="") as fh:
            return int(next(csv.DictReader(fh))["size"])
    except (KeyError, OSError, StopIteration, ValueError, TypeError):
        return None


def tiled_forward(model, x, tile: int):
    """Run `model` on non-overlapping tile x tile windows of `x` and stitch the logits.

    Lets a model trained on 512 crops be scored on exactly the pixels of a
    1024-crop val set while still seeing 512 inputs, as it did in training --
    the only way to compare input window sizes on identical pixels. The tiles
    do not overlap, so each one is precisely a 512 crop the model could have
    been validated on; overlap-averaging would hand it extra context instead.
    """
    b, c, h, w = x.shape
    if h % tile or w % tile:
        raise ValueError(f"input {h}x{w} is not a multiple of tile {tile}")
    nh, nw = h // tile, w // tile
    tiles = x.reshape(b, c, nh, tile, nw, tile).permute(0, 2, 4, 1, 3, 5).reshape(b * nh * nw, c, tile, tile)
    out = model(tiles)
    k = out.shape[1]
    return out.reshape(b, nh, nw, k, tile, tile).permute(0, 3, 1, 4, 2, 5).reshape(b, k, h, w)


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
    tile_mode: str = "off",
) -> dict:
    model, model_kind, num_classes, config = load_segmentation_model_from_checkpoint(
        ckpt_path, device=str(device)
    )
    tile = trained_crop_size(config) if tile_mode == "auto" else None
    if tile_mode == "auto" and tile is None:
        print(f"  [tile] {ckpt_path.name}: training crop size unknown -- evaluating untiled")

    # The value the training run itself recorded at the epoch this checkpoint
    # was saved. Read it from the checkpoint, NOT from config["output"]
    # ["history_path"]: that path is resolved at eval time and, with sweeps
    # writing several trials, routinely points at another run's CSV or at a
    # stale file -- it reported 0.0739 for a checkpoint whose own metrics say
    # 0.1690. Since the training loop now accumulates a global confusion
    # matrix, this should agree with global_miou below; a disagreement means
    # the checkpoint predates that change.
    reported_val_miou = None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        reported_val_miou = float(ckpt.get("metrics", {}).get("miou"))
        del ckpt
    except Exception:
        pass

    if num_classes_from_config is not None and num_classes != num_classes_from_config:
        raise ValueError(
            f"{ckpt_path}: checkpoint num_classes={num_classes} != loader "
            f"num_classes={num_classes_from_config}"
        )

    # One global confusion matrix (rows = truth, cols = prediction) rather than
    # per-class masks. It yields intersection/union for free, costs one kernel
    # per batch instead of 4*num_classes with a .item() on each, and -- the
    # reason it is here -- it says WHAT a class is confused with, which is what
    # decides whether a weak class needs a different loss, more capacity, or
    # better labels.
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    tiled = False

    with torch.no_grad():
        for batch in dataloader:
            local, target, context = parse_segmentation_batch(batch)
            local = local.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()
            context_tensor = context.to(device, non_blocking=True).float() if context is not None else None

            if tile is not None and local.shape[-1] > tile:
                if context_tensor is not None:
                    raise ValueError(f"{ckpt_path.name}: tiling a context model is not supported")
                logits = tiled_forward(model, local, tile)
                tiled = True
            elif context_tensor is not None:
                logits = model(local, context_tensor)
            else:
                logits = model(local)
            preds = logits.argmax(dim=1)
            in_range = (target != ignore_index) & (target >= 0) & (target < num_classes)
            t = target[in_range].reshape(-1)
            p = preds[in_range].reshape(-1)
            conf += torch.bincount(
                t * num_classes + p, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    conf = conf.cpu()
    intersection = conf.diagonal().clone()
    union = conf.sum(0) + conf.sum(1) - intersection
    correct_total = int(intersection.sum())
    valid_total = int(conf.sum())

    present = union > 0
    per_class_iou = torch.where(
        present, intersection.double() / union.clamp(min=1).double(), torch.zeros(num_classes, dtype=torch.float64)
    )
    global_miou = float(per_class_iou[present].mean()) if present.any() else 0.0
    global_pixel_acc = correct_total / max(valid_total, 1)

    return {
        "ckpt_path": ckpt_path,
        "tile": tile if tiled else None,
        "num_classes": num_classes,
        "reported_val_miou": reported_val_miou,
        "global_miou": global_miou,
        "global_pixel_acc": global_pixel_acc,
        "present": present,
        "per_class_iou": per_class_iou,
        "intersection": intersection,
        "union": union,
        "confusion": conf,
    }


def print_checkpoint_report(result: dict, *, split: str) -> None:
    ckpt_path = result["ckpt_path"]
    num_classes = result["num_classes"]
    present = result["present"]
    per_class_iou = result["per_class_iou"]

    print(f"=== {ckpt_path.parent.name}/{ckpt_path.name} ===")
    if result.get("tile"):
        print(f"  tiled: non-overlapping {result['tile']}x{result['tile']} windows (its training crop size)")
    if result["reported_val_miou"] is not None:
        print(f"  val_miou recorded by the run at this checkpoint's epoch: {result['reported_val_miou']:.4f}")
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


def print_confusions(result: dict, *, max_classes: int = 6, min_share: float = 0.10) -> None:
    """For the weakest classes, where does their truth actually get sent?

    A class can score a low IoU for opposite reasons -- its pixels are handed
    to one specific neighbour (a discrimination problem, fixable with loss or
    capacity), or they are scattered (a label-quality or feature problem). The
    prescription differs, so print the split rather than just the score.
    """
    conf = result.get("confusion")
    if conf is None:
        return
    per_class_iou = result["per_class_iou"]
    present = result["present"]
    order = sorted(
        (i for i in range(result["num_classes"]) if bool(present[i])),
        key=lambda i: float(per_class_iou[i]),
    )[:max_classes]
    print("  Where each weak class's true pixels actually go:")
    for i in order:
        row = conf[i]
        total = int(row.sum())
        if total == 0:
            continue
        recall = int(row[i]) / total
        parts = []
        for j in sorted(range(len(row)), key=lambda j: int(row[j]), reverse=True):
            share = int(row[j]) / total
            if share < min_share or j == i:
                continue
            parts.append(f"{share*100:.0f}% -> {class_name(j)}")
            if len(parts) == 3:
                break
        print(f"    {class_name(i):<52s} IoU={float(per_class_iou[i]):.4f} "
              f"recall={recall*100:5.1f}%  {'; '.join(parts) if parts else '(scattered)'}")
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
    parser.add_argument(
        "--context-cache-dir", default=None,
        help="Context crop cache (prep_seg_context_cache.py output). Required to\n"
             "evaluate a checkpoint whose model was built with use_context=True; "
             "without it the loader falls back to a slow live per-item read.",
    )
    parser.add_argument(
        "--crop-cache-dir", default=None,
        help="Padded crop cache matching --loader-config-path's manifest (as passed\n"
             "to training). Same pixels as the live raster read, minus the per-crop\n"
             "GeoTIFF decompression.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument(
        "--tile", choices=("off", "auto"), default="off",
        help="auto: a checkpoint trained on smaller crops than the loader serves\n"
             "(e.g. a 512 model on the 1024 val set) is run on non-overlapping\n"
             "windows of its training crop size and the logits stitched, so models\n"
             "of different input sizes are scored on identical pixels.",
    )
    parser.add_argument("--json-out", default=None,
                        help="Also write per-checkpoint mIoU and per-class IoU here (JSON).")
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
    if args.crop_cache_dir:
        loader_kwargs["cache_dir"] = args.crop_cache_dir
    num_classes_from_config = loader_kwargs.get("num_classes")

    # A context-branch model needs the loader to SERVE context crops. Building
    # one context-free loader for every checkpoint made the context variants
    # (v1/v3) die with "built with use_context=True but got context_x=None",
    # i.e. this evaluator could not score half the variant comparison. Whether
    # context is needed is a property of each checkpoint, so build (and reuse)
    # one loader per configuration rather than assuming.
    _loader_cache: dict[bool, tuple] = {}

    def _loader_for(use_context: bool):
        if use_context not in _loader_cache:
            kwargs = dict(loader_kwargs)
            if use_context:
                kwargs["use_context"] = True
                if args.context_cache_dir:
                    kwargs["context_cache_dir"] = args.context_cache_dir
            bundle = load_loader_bundle(
                "vision_backend.seg_dataset:create_segmentation_dataloaders", kwargs
            )
            ds = bundle.get(f"{args.split}_dataset")
            n = len(ds.records) if ds is not None and hasattr(ds, "records") else "?"
            print(f"Split: {args.split} ({n} crops, context={'on' if use_context else 'off'})\n")
            _loader_cache[use_context] = bundle[args.split]
        return _loader_cache[use_context]

    def _needs_context(ckpt_path_str: str) -> bool:
        try:
            ck = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
            return bool(ck.get("config", {}).get("model", {}).get("use_context", False))
        except Exception:
            return False

    def _evaluate(ckpt_path_str: str) -> dict:
        return evaluate_checkpoint(
            Path(ckpt_path_str),
            dataloader=_loader_for(_needs_context(ckpt_path_str)),
            ignore_index=args.ignore_index,
            device=device,
            num_classes_from_config=num_classes_from_config,
            torch=torch,
            load_segmentation_model_from_checkpoint=load_segmentation_model_from_checkpoint,
            parse_segmentation_batch=parse_segmentation_batch,
            resolve_path=resolve_path,
            tile_mode=args.tile,
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
        print_confusions(result)
        if baseline_result is not None:
            print_delta_table(baseline_result, result)

    if args.json_out:
        rows = [
            {
                "checkpoint": path,
                "tile": r["tile"],
                "reported_val_miou": r["reported_val_miou"],
                "global_miou": r["global_miou"],
                "global_pixel_acc": r["global_pixel_acc"],
                "per_class_iou": {
                    class_name(c): float(r["per_class_iou"][c])
                    for c in range(r["num_classes"]) if bool(r["present"][c])
                },
            }
            for path, r in results_by_path.items()
        ]
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
