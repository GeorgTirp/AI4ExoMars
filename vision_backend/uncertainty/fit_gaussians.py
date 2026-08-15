#!/usr/bin/env python3
"""Fit per-class Mahalanobis Gaussians from a trained segmentation checkpoint.

Offline calibration step: scans labeled crops (the same NOAH-H manifest format
`seg_dataset.py` uses for stage-3 training), extracts each pixel's feature vector
(from `model.features.extract_pixel_features`) together with its ground-truth
class, and fits one Gaussian per class (or a shared covariance across classes --
the standard, more stable choice). The resulting `MahalanobisStats` is saved next
to the checkpoint as `<checkpoint_stem>.uncertainty.pt`, where MarsObsLabeling's
`mars-inference` Uncertainty Heatmap button expects to find it.

Fit on the *training* split (default): Mahalanobis OOD scoring measures distance
from what the model has seen, so fitting on held-out data would defeat the point.

Example
-------
python3 vision_backend/uncertainty/fit_gaussians.py \\
  --checkpoint checkpoints/stage3_segmentation.pt \\
  --manifest-path data/seg_crops/manifest.csv \\
  --imagery-path data/seg_crops/imagery.tif \\
  --label-path data/seg_crops/labels.tif
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    from vision_backend.seg_dataset import (
        IGNORE_INDEX,
        SegmentationCropDataset,
        load_seg_records,
        partition_records,
    )
    from vision_backend.training.builders import load_segmentation_model_from_checkpoint
    from vision_backend.training.utils import select_device
    from vision_backend.uncertainty.malahanobis import fit_class_gaussians, save_stats
except ModuleNotFoundError:
    from seg_dataset import IGNORE_INDEX, SegmentationCropDataset, load_seg_records, partition_records
    from training.builders import load_segmentation_model_from_checkpoint
    from training.utils import select_device
    from uncertainty.malahanobis import fit_class_gaussians, save_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Trained stage-3 segmentation checkpoint (.pt)")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--imagery-path", required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX)
    parser.add_argument("--pixels-per-crop-per-class", type=int, default=50,
                        help="Random pixel subsample per class, per crop (bounds memory).")
    parser.add_argument("--max-samples-per-class", type=int, default=5000,
                        help="Stop accumulating a class once it reaches this many pixels.")
    parser.add_argument("--shared-covariance", dest="shared_covariance", action="store_true", default=True)
    parser.add_argument("--per-class-covariance", dest="shared_covariance", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output path (default: <checkpoint_stem>.uncertainty.pt next to the checkpoint)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    import torch

    device = select_device(torch) if args.device == "auto" else torch.device(args.device)
    model, model_kind, num_classes, _ = load_segmentation_model_from_checkpoint(
        args.checkpoint, device=device
    )
    print(f"Loaded {model_kind} model, {num_classes} classes, device={device}")

    records = load_seg_records(args.manifest_path)
    if args.split != "all":
        train_records, val_records = partition_records(records)
        records = train_records if args.split == "train" else val_records
    print(f"{len(records)} crops in split={args.split!r}")

    dataset = SegmentationCropDataset(
        records,
        imagery_path=args.imagery_path,
        label_path=args.label_path,
        augment=False,
        ignore_index=args.ignore_index,
    )

    features_by_class: dict[int, list[torch.Tensor]] = {c: [] for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}

    try:
        from vision_backend.model.features import extract_pixel_features
    except ModuleNotFoundError:
        from model.features import extract_pixel_features

    with torch.no_grad():
        for i in range(len(dataset)):
            if all(counts[c] >= args.max_samples_per_class for c in range(num_classes)):
                print("All classes reached max-samples-per-class; stopping early.")
                break

            sample = dataset[i]
            image = sample["image"].unsqueeze(0).to(device)  # [1,1,H,W]
            label = sample["label"]  # [H,W], long, ignore_index for nodata/boundary

            feats = extract_pixel_features(model, image)[0]  # [F,H,W]
            f, h, w = feats.shape
            feats_flat = feats.permute(1, 2, 0).reshape(-1, f)  # [H*W, F]
            labels_flat = label.reshape(-1)

            for c in range(num_classes):
                if counts[c] >= args.max_samples_per_class:
                    continue
                idx = (labels_flat == c).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                take = min(args.pixels_per_crop_per_class, idx.numel())
                chosen = idx[torch.randperm(idx.numel())[:take]]
                features_by_class[c].append(feats_flat[chosen].cpu())
                counts[c] += take

            if (i + 1) % 50 == 0:
                print(f"  scanned {i + 1}/{len(dataset)} crops -- counts={counts}")

    print(f"Final per-class pixel counts: {counts}")
    all_features = torch.cat([torch.cat(v) for v in features_by_class.values() if v], dim=0)
    all_labels = torch.cat(
        [torch.full((sum(t.shape[0] for t in v),), c, dtype=torch.long) for c, v in features_by_class.items() if v]
    )

    stats = fit_class_gaussians(all_features, all_labels, shared_covariance=args.shared_covariance)

    output = Path(args.output) if args.output else Path(args.checkpoint).with_suffix("").with_suffix(".uncertainty.pt")
    save_stats(stats, output)
    print(f"Saved Mahalanobis stats ({len(stats.class_stats)} classes, feature_dim={stats.feature_dim}) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
