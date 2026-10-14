#!/usr/bin/env python3
"""Campaign posterior of a block classifier: the next campaign's prior.

Diagonal Laplace (online / sequential Bayes; Ritter et al. 2018, Sliwa et al.,
arXiv:2410.06800): precision = tau * F_prior + N * F_expert at the fine-tuned
weights, over all BlockClassifier parameters, where F_prior is the prior the
run was trained with (its --laplace-prior / --laplace-tau) and F_expert the
per-block MC Fisher on the run's training blocks
(laplace_prior.mc_fisher_diagonal_blocks). Fine-tune the next campaign with
--init-block-checkpoint <this checkpoint> --laplace-prior <out> --laplace-tau 1.

    python scripts/fit_campaign_posterior.py checkpoints/block_finetune/<run>/best.pt --out <posterior.pt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_block_artifacts import fit_posterior  # noqa: E402
from vision_backend.block_dataset import DEFAULT_DC_CLASSES_JSON, load_block_manifest  # noqa: E402
from vision_backend.train_stage4_block_finetune import load_block_classifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_block_classifier(args.checkpoint, device=str(device))
    cargs = ckpt["args"]
    records, classes = load_block_manifest(cargs["manifest"], cargs.get("dc_classes_json") or DEFAULT_DC_CLASSES_JSON,
                                           cargs.get("block_classes"))
    train = [r for r in records if r.split == "train"]
    base = int(ckpt["head_config"].get("tile_size", 512))
    summary = fit_posterior(model, ckpt, train, args, device, Path(args.out), base)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
