#!/usr/bin/env python3
"""Pool the spatial-CV fold evaluations of E18 (scripts/eval_link_cv_fold.py)
and compare the output links, paired on the same held-out blocks.

For each link and predictive: accuracy, macro-F1 (classes with >= 5 blocks),
NLL, ECE and Brier over all held-out blocks of the 3 folds, plus the spread
over folds. Uncertainty scores: AUROC for flagging misclassified blocks
(held-out) and for flagging the Oxia NOAH-H crops against the held-out blocks
(mean over the 3 folds x 2 cross-fitting halves). Differences normcdf -
softmax with a paired bootstrap over 1 km cells.

    python scripts/summarize_link_cv.py --links softmax normcdf --in-dir results/block_link_cv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_block_artifacts import auroc  # noqa: E402
from vision_backend.block_dataset import BlockClasses  # noqa: E402
from vision_backend.train_stage4_block_finetune import summarize  # noqa: E402

PREDICTIVES = ("map", "map_temperature", "laplace_closed", "laplace_mc")
KEYS = ("accuracy", "macro_f1_sel", "nll", "ece", "brier")


def metrics(p: np.ndarray, y: np.ndarray, classes) -> dict:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    m = summarize(np.log(p), y, classes, min_support=5)
    onehot = np.eye(p.shape[1])[y]
    return {"accuracy": m["accuracy"], "macro_f1_sel": m["macro_f1_sel"], "nll": m["nll"], "ece": m["ece"],
            "brier": float(((p - onehot) ** 2).sum(1).mean())}


def load(in_dir: Path, link: str, k: int) -> dict:
    folds = [dict(np.load(in_dir / f"{link}.fold{f}.npz", allow_pickle=False)) for f in range(k)]
    info = [json.loads((in_dir / f"{link}.fold{f}.json").read_text()) for f in range(k)]
    pooled = {key: np.concatenate([d[key] for d in folds]) for key in folds[0]
              if not key.startswith("oxia_") and not key.startswith("eval_half")}
    pooled["fold"] = np.concatenate([np.full(len(d["label"]), f) for f, d in enumerate(folds)])
    order = np.argsort(pooled["block_id"])
    pooled = {key: v[order] for key, v in pooled.items()}
    return {"pooled": pooled, "folds": folds, "info": info}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--links", nargs="+", default=["softmax", "normcdf"])
    ap.add_argument("--in-dir", default="results/block_link_cv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="default: <in-dir>/summary.json")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    runs = {link: load(in_dir, link, args.k) for link in args.links}
    names = runs[args.links[0]]["info"][0]["classes"]
    classes = BlockClasses(names=names, seg_index=[None] * len(names))

    ref = runs[args.links[0]]["pooled"]
    for link, r in runs.items():
        if not np.array_equal(r["pooled"]["block_id"], ref["block_id"]):
            raise SystemExit(f"{link}: held-out blocks differ from {args.links[0]}'s")
    y, cells = ref["label"], ref["cell"]

    out = {"n_heldout_blocks": int(len(y)), "links": {}}
    for link, r in runs.items():
        p = r["pooled"]
        res = {"predictive": {}, "misclassification_auroc": {}, "oxia_auroc": {}, "tuned": [i["halves"] for i in r["info"]]}
        for pred in PREDICTIVES:
            pooled = metrics(p[f"p_{pred}"], y, classes)
            per_fold = [metrics(p[f"p_{pred}"][p["fold"] == f], y[p["fold"] == f], classes) for f in range(args.k)]
            res["predictive"][pred] = pooled | {"fold_sd": {k: float(np.std([m[k] for m in per_fold], ddof=1))
                                                          for k in KEYS}}
        wrong = p["p_map"].argmax(1) != y
        score_keys = sorted(k for k in p if k.startswith("s_"))
        for key in score_keys:
            res["misclassification_auroc"][key[2:]] = auroc(p[key], wrong)
        for key in score_keys:
            vals = []
            for d in r["folds"]:
                for half in (0, 1):
                    ox = d.get(f"oxia_{key}_half{half}")
                    if ox is None:
                        continue
                    ev = d[f"eval_half{half}"].astype(bool)
                    held = d[key][ev]
                    vals.append(auroc(np.r_[ox, held], np.r_[np.ones(len(ox)), np.zeros(len(held))]))
            if vals:
                res["oxia_auroc"][key[2:]] = float(np.mean(vals))
        out["links"][link] = res

    # paired differences (second link - first), bootstrap over cells
    if len(args.links) == 2:
        a, b = (runs[link]["pooled"] for link in args.links)
        uniq = np.unique(cells)
        members = {c: np.flatnonzero(cells == c) for c in uniq}
        rng = np.random.default_rng(0)
        boots = [np.concatenate([members[c] for c in rng.choice(uniq, size=len(uniq))]) for _ in range(args.n_boot)]
        diff = {}
        for pred in PREDICTIVES:
            point = {k: metrics(b[f"p_{pred}"], y, classes)[k] - metrics(a[f"p_{pred}"], y, classes)[k] for k in KEYS}
            draws = np.array([[metrics(b[f"p_{pred}"][i], y[i], classes)[k] - metrics(a[f"p_{pred}"][i], y[i], classes)[k]
                               for k in KEYS] for i in boots])
            lo, hi = np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)
            diff[pred] = {k: {"diff": float(point[k]), "ci95": [float(l_), float(h_)]}
                          for k, l_, h_ in zip(KEYS, lo, hi)}
        out["paired_difference"] = {"of": f"{args.links[1]} - {args.links[0]}", "by_predictive": diff}

    path = Path(args.out) if args.out else in_dir / "summary.json"
    path.write_text(json.dumps(out, indent=2))
    for link, res in out["links"].items():
        print(f"== {link}")
        for pred, m in res["predictive"].items():
            print(f"  {pred:<17} " + "  ".join(f"{k} {m[k]:.4f}" for k in KEYS))
        print("  misclassification AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in res["misclassification_auroc"].items()))
        print("  Oxia AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in res["oxia_auroc"].items()))
    if "paired_difference" in out:
        print(f"== paired {out['paired_difference']['of']} (95 % CI over cells)")
        for pred, d in out["paired_difference"]["by_predictive"].items():
            print(f"  {pred:<17} " + "  ".join(f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                                                for k, v in d.items()))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
