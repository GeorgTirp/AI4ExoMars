#!/usr/bin/env python3
"""Build a Neural PCA explainability gallery from a trained segmentation checkpoint.

Offline calibration step: scans labeled crops, computes each class's psi_k(x) =
w_k (elementwise*) phi(x) embedding (see `pc_align/neural_pca.py`), fits PCA per
class, and saves a portable gallery of the top-activating crop *thumbnails* per
(class, component) -- self-contained (no need for the training corpus to be
present later), so MarsObsLabeling's class Summary window can just load and
display it. Saved next to the checkpoint as `<checkpoint_stem>.npca.pt`.

Example
-------
python3 vision_backend/pc_align/fit_neural_pca.py \\
  --checkpoint checkpoints/stage3_segmentation.pt \\
  --manifest-path data/seg_crops/manifest.csv \\
  --imagery-path data/seg_crops/imagery.tif \\
  --label-path data/seg_crops/labels.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from vision_backend.pc_align.neural_pca import (
        build_gallery_entry,
        compute_psi_batch,
        fit_class_pca,
        save_gallery,
    )
    from vision_backend.seg_dataset import IGNORE_INDEX, SegmentationCropDataset, load_seg_records, partition_records
    from vision_backend.training.builders import load_segmentation_model_from_checkpoint
    from vision_backend.training.utils import select_device
except ModuleNotFoundError:
    from pc_align.neural_pca import build_gallery_entry, compute_psi_batch, fit_class_pca, save_gallery
    from seg_dataset import IGNORE_INDEX, SegmentationCropDataset, load_seg_records, partition_records
    from training.builders import load_segmentation_model_from_checkpoint
    from training.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--imagery-path", required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX)
    parser.add_argument("--presence-fraction", type=float, default=0.05,
                        help="A crop 'contains' a class if at least this fraction of its "
                             "pixels carry that label.")
    parser.add_argument("--max-samples-per-class", type=int, default=200)
    parser.add_argument("--min-samples-per-class", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=4, help="The '4 orthogonal features'.")
    parser.add_argument("--top-k", type=int, default=6, help="Gallery items kept per component.")
    parser.add_argument("--thumbnail-size", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <checkpoint_stem>.npca.pt next to the checkpoint)")
    return parser.parse_args()


def _thumbnail(image_tensor, size: int):
    """[1,H,W] tensor in [-1,1] -> uint8 [size,size] numpy array."""
    import numpy as np
    import torch.nn.functional as F

    resized = F.interpolate(
        image_tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    )[0, 0]
    as_uint8 = ((resized.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).byte()
    return as_uint8.cpu().numpy().astype(np.uint8)


def main() -> int:
    args = parse_args()

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

    print("Scanning dataset for class presence...")
    class_to_indices: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        label = dataset[i]["label"]
        valid = label != args.ignore_index
        total = int(valid.sum())
        if total == 0:
            continue
        for c in range(num_classes):
            frac = float((label == c).sum()) / total
            if frac >= args.presence_fraction:
                class_to_indices[c].append(i)

    gallery: dict[int, dict[int, list]] = {}

    with torch.no_grad():
        for class_id, indices in class_to_indices.items():
            if len(indices) < args.min_samples_per_class:
                print(f"class {class_id}: {len(indices)} samples < min_samples_per_class, skipping")
                continue
            indices = indices[: args.max_samples_per_class]
            print(f"class {class_id}: using {len(indices)} samples")

            psi_rows = []
            thumbnails = []
            source_ids = []
            for idx in indices:
                sample = dataset[idx]
                image = sample["image"].unsqueeze(0).to(device)  # [1,1,H,W]
                psi = compute_psi_batch(model, class_id, image)[0].cpu()
                psi_rows.append(psi)
                thumbnails.append(_thumbnail(sample["image"], args.thumbnail_size))
                record = records[idx]
                source_ids.append(f"{Path(args.imagery_path).stem}_{record.col}_{record.row}")

            psi = torch.stack(psi_rows)
            pca = fit_class_pca(psi, class_id=class_id, n_components=args.n_components)
            gallery[class_id] = build_gallery_entry(
                pca, psi, thumbnails, source_ids, top_k=args.top_k
            )

    output = Path(args.output) if args.output else Path(args.checkpoint).with_suffix("").with_suffix(".npca.pt")
    save_gallery(gallery, output)
    print(f"Saved neural-PCA gallery ({len(gallery)} classes) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
