#!/usr/bin/env python3
"""Ensembling and flip test-time augmentation on the 512 val set.

Every listed checkpoint is run once per flip view (identity, horizontal,
vertical, both -- the augmentations the models were trained with; rotations
are not, they would change the illumination direction), predictions are
flipped back, and each named configuration averages the softmax
probabilities of its members (and views). Reported per configuration over all
labelled val pixels: global mIoU, pixel accuracy, NLL, Brier, ECE, AUROC, and
per-class IoU.

    python scripts/eval_ensemble.py --baseline A.pt --members B.pt C.pt D.pt \
        --loader-config-path ... --crop-cache-dir ... --json-out out.json
The first member is treated as the best single model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.eval_global_miou import class_name
from scripts.eval_uncertainty import Acc
from vision_backend.training.builders import load_segmentation_model_from_checkpoint
from vision_backend.training.utils import load_loader_bundle, parse_segmentation_batch, select_device

VIEWS = ((), (-1,), (-2,), (-2, -1))  # dims to flip: none, W, H, both


def probs_per_view(model, x):
    out = []
    for dims in VIEWS:
        xi = x.flip(dims) if dims else x
        p = F.softmax(model(xi).float(), dim=1)
        out.append(p.flip(dims) if dims else p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--loader-config-path", required=True)
    ap.add_argument("--crop-cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    device = select_device(torch)
    kw = json.loads(Path(args.loader_config_path).read_text())
    kw.update(batch_size=args.batch_size, num_workers=args.num_workers, augment=False)
    if args.crop_cache_dir:
        kw["cache_dir"] = args.crop_cache_dir
    loader = load_loader_bundle("vision_backend.seg_dataset:create_segmentation_dataloaders", kw)["val"]

    paths = [args.baseline] + args.members
    models = [load_segmentation_model_from_checkpoint(Path(p), device=str(device))[0] for p in paths]
    c = models[0].decoder.head.out_channels
    base, best, members = 0, 1, list(range(1, len(paths)))
    k = len(members)
    configs = {
        "baseline (NAdamW, 30 ep)": ([base], False),
        "best single": ([best], False),
        "best single + flip TTA": ([best], True),
        f"top-{k} ensemble": (members, False),
        f"top-{k} ensemble + flip TTA": (members, True),
    }
    accs = {name: Acc(c, device) for name in configs}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            x, y, _ = parse_segmentation_batch(batch)
            x, y = x.to(device).float(), y.to(device).long()
            views = [probs_per_view(m, x) for m in models]  # [model][view] -> [B, C, H, W]
            yy = y.reshape(-1)
            keep = yy != args.ignore_index
            for name, (idx, tta) in configs.items():
                ps = [views[j][v] for j in idx for v in (range(len(VIEWS)) if tta else [0])]
                p = torch.stack(ps).mean(0).permute(0, 2, 3, 1).reshape(-1, c)[keep]
                accs[name].add(torch.log(p.clamp_min(1e-12)), yy[keep])

    rows = {}
    print(f"{'configuration':32s} {'mIoU':>7s} {'acc':>7s} {'NLL':>7s} {'ECE':>7s} {'AUROC(1-maxp)':>14s}")
    for name, a in accs.items():
        s = a.summary()
        inter = a.conf.diagonal()
        union = a.conf.sum(0) + a.conf.sum(1) - inter
        s["per_class_iou"] = {class_name(j): float(inter[j] / union[j]) for j in range(c) if union[j] > 0}
        rows[name] = s
        print(f"{name:32s} {s['miou']:7.4f} {s['pixel_acc']:7.4f} {s['nll']:7.4f} {s['ece']:7.4f} {s['auroc_1-maxprob']:14.3f}")

    show = ["baseline (NAdamW, 30 ep)", "best single", f"top-{k} ensemble + flip TTA"]
    print(f"\n{'class':52s}" + "".join(f"{n.split(' (')[0][:16]:>18s}" for n in show))
    classes = sorted(rows[show[0]]["per_class_iou"], key=lambda cl: -rows[show[0]]["per_class_iou"][cl])
    for cl in classes:
        print(f"{cl[:52]:52s}" + "".join(f"{rows[n]['per_class_iou'].get(cl, float('nan')):18.3f}" for n in show))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps({"checkpoints": paths, "results": rows}, indent=1))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
