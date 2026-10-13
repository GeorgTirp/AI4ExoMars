#!/usr/bin/env python3
"""Paired comparison of Stage-4 runs on the same test blocks.

Single-run test CIs are wide (E13: accuracy 0.656 [0.52, 0.77]) because the
test set is 355 spatially clustered blocks -- but two runs are scored on the
SAME blocks, so their difference is far less noisy than either number. This
resamples whole spatial cells (identically for both runs) and reports the
difference run - reference with a 95 % CI, for accuracy, macro-F1 (classes
with >= 5 test blocks) and the temperature-scaled NLL (each run's own T*),
plus the NOAH-H val mIoU each run kept (eval.json, no CI).

    python scripts/compare_block_runs.py checkpoints/block_finetune/experts_v1_plain \
        checkpoints/block_finetune/<run> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend.block_dataset import load_block_classes  # noqa: E402
from vision_backend.train_stage4_block_finetune import summarize  # noqa: E402


def load(run_dir: Path) -> dict:
    p = np.load(run_dir / "eval_preds.npz")
    ev = json.loads((run_dir / "eval.json").read_text())
    return {"z": p["test_logits"] / float(p["temperature"]), "y": p["test_labels"], "g": p["test_cells"],
            "noahh": ev.get("noahh_val_miou")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--block-classes", default="data/BlockLabels_prepared/block_classes.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    classes = load_block_classes(args.block_classes)
    ref = load(Path(args.reference))
    cells = np.unique(ref["g"])
    members = {c: np.flatnonzero(ref["g"] == c) for c in cells}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[c] for c in rng.choice(cells, size=len(cells))]) for _ in range(args.n_boot)]
    keys = ("accuracy", "macro_f1_sel", "nll")

    def metrics(r, idx):
        m = summarize(r["z"][idx], r["y"][idx], classes, min_support=args.min_support)
        return [m[k] for k in keys]

    all_idx = np.arange(len(ref["y"]))
    base = metrics(ref, all_idx)
    print(f"reference {args.reference}: " + "  ".join(f"{k} {v:.4f}" for k, v in zip(keys, base))
          + f"  NOAH-H mIoU {ref['noahh']}")
    out = {"reference": args.reference, "reference_metrics": dict(zip(keys, base)), "runs": {}}
    for run in args.runs:
        r = load(Path(run))
        if not (np.array_equal(r["y"], ref["y"]) and np.array_equal(r["g"], ref["g"])):
            raise SystemExit(f"{run}: test blocks differ from the reference's")
        point = np.array(metrics(r, all_idx)) - np.array(base)
        diffs = np.array([np.array(metrics(r, i)) - np.array(metrics(ref, i)) for i in boots])
        lo, hi = np.nanpercentile(diffs, 2.5, axis=0), np.nanpercentile(diffs, 97.5, axis=0)
        row = {k: {"diff": float(d), "ci95": [float(a), float(b)]} for k, d, a, b in zip(keys, point, lo, hi)}
        row["noahh_val_miou"] = r["noahh"]
        out["runs"][run] = row
        print(f"{run}:")
        for k, d, a, b in zip(keys, point, lo, hi):
            flag = "  *" if (a > 0 or b < 0) else ""
            print(f"    {k:<13} {d:+.4f}  [{a:+.4f}, {b:+.4f}]{flag}")
        if r["noahh"] is not None and ref["noahh"] is not None:
            print(f"    NOAH-H mIoU   {r['noahh']:.4f} (reference {ref['noahh']:.4f}, before any fine-tuning 0.2072)")
    print("(* = 95 % CI excludes 0; NLL: lower is better)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
