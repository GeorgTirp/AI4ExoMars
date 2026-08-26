#!/usr/bin/env python3
"""Collect the model-variant sweep results into variant_comparison.md.

Reads the JSONL rows each run appends via --variant-metrics-path and reports the
BEST-OF-N per variant (best = highest val/miou), so a variant is judged by what
it achieved with a properly tuned lr/weight_decay rather than by its average
over a small Bayesian budget.

Runs after the sweeps finish; it never touches wandb or the GPU, so it is safe
to re-run at any time (including while runs are still in flight -- it just
reports fewer rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

# Fixed order so the table always reads reference-first, and a missing variant
# is visible as an explicit gap rather than silently absent.
VARIANT_ORDER = ["v0", "v2", "v1", "v3"]
VARIANT_LABEL = {
    "v0": "V0  big / no context",
    "v2": "V2  small / no context",
    "v1": "V1  big / context",
    "v3": "V3  small / context",
}


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[collect] skipping malformed line in {path}")
    return rows


def best_per_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        vid = row.get("variant_id")
        if not vid:
            continue
        score = row.get("best_val_miou")
        if score is None:
            continue
        current = best.get(vid)
        if current is None or score > current.get("best_val_miou", float("-inf")):
            best[vid] = row
    return best


def _fmt(value: Optional[float], spec: str = "{:.4f}", scale: float = 1.0) -> str:
    if value is None:
        return "--"
    try:
        return spec.format(float(value) * scale)
    except (TypeError, ValueError):
        return str(value)


def _ratio(small: Optional[float], big: Optional[float]) -> str:
    if not small or not big:
        return "--"
    return f"{small / big:.3f}x"


def render(rows: list[dict[str, Any]], counts: dict[str, int]) -> str:
    best = best_per_variant(rows)
    lines: list[str] = []
    lines.append("# Model-variant comparison\n")
    lines.append(
        "Best-of-N per variant (N = runs completed), best = highest `val/miou`. "
        "Everything except the architecture is held constant across variants; "
        "only `{width, use_context}` differ between them and only "
        "`{learning_rate, weight_decay}` were searched within one.\n"
    )

    lines.append("| variant | runs | val/miou (DC) | val/miou (IG) | params total | params enc | "
                 "step ms | infer img/s | peak GB |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for vid in VARIANT_ORDER:
        row = best.get(vid)
        label = VARIANT_LABEL.get(vid, vid)
        if row is None:
            lines.append(f"| {label} | 0 | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {label} | {counts.get(vid, 0)} | {_fmt(row.get('best_val_miou'))} | "
            f"{_fmt(row.get('best_val_miou_ig'))} | "
            f"{_fmt(row.get('params_total'), '{:.2f} M', 1e-6)} | "
            f"{_fmt(row.get('params_encoder'), '{:.2f} M', 1e-6)} | "
            f"{_fmt(row.get('train_step_ms'), '{:.0f}')} | "
            f"{_fmt(row.get('infer_images_per_s'), '{:.2f}')} | "
            f"{_fmt(row.get('peak_mem_gb'), '{:.2f}')} |"
        )

    # small-vs-big, the question the width axis exists to answer
    lines.append("\n## Small vs big\n")
    lines.append("| pair | params | step time | val/miou delta |")
    lines.append("|---|---|---|---|")
    for big_id, small_id, tag in (("v0", "v2", "no context"), ("v1", "v3", "context")):
        b, s = best.get(big_id), best.get(small_id)
        if not b or not s:
            lines.append(f"| {small_id}/{big_id} ({tag}) | -- | -- | -- |")
            continue
        d = s.get("best_val_miou", 0) - b.get("best_val_miou", 0)
        lines.append(
            f"| {small_id}/{big_id} ({tag}) | "
            f"{_ratio(s.get('params_total'), b.get('params_total'))} | "
            f"{_ratio(s.get('train_step_ms'), b.get('train_step_ms'))} | {d:+.4f} |"
        )

    lines.append("\n## Context on vs off\n")
    lines.append("| pair | val/miou delta | params delta |")
    lines.append("|---|---|---|")
    for off_id, on_id, tag in (("v0", "v1", "big"), ("v2", "v3", "small")):
        off, on = best.get(off_id), best.get(on_id)
        if not off or not on:
            lines.append(f"| {on_id} vs {off_id} ({tag}) | -- | -- |")
            continue
        d = on.get("best_val_miou", 0) - off.get("best_val_miou", 0)
        dp = (on.get("params_total") or 0) - (off.get("params_total") or 0)
        lines.append(f"| {on_id} vs {off_id} ({tag}) | {d:+.4f} | {dp / 1e6:+.2f} M |")

    lines.append(
        "\n## Reading these numbers\n\n"
        "- **The small variants narrow the DECODER as well as the encoder.** The\n"
        "  fixed-width decoder is ~73% of a training step, so encoder width alone\n"
        "  moved wall-clock by only ~3%; `decoder_channels` was scaled with it to\n"
        "  make the time axis meaningful. V0-vs-V2 is therefore a whole-model width\n"
        "  comparison, not an encoder-only one.\n"
        "- **V1/V3 are trained from scratch, which undersells the context branch.**\n"
        "  Its payoff normally needs pretraining, so treat these as a plumbing and\n"
        "  relative read -- not a verdict on whether context helps. The comparison\n"
        "  that would settle that is the pretrained rerun (MODEL_VARIANTS_EXPERIMENT.md §9).\n"
        "- Runs counted per variant reflect only rows written; a crashed run\n"
        "  contributes nothing and shows up as a lower `runs` count.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", default="results/variant_comparison",
                    help="Directory holding the per-variant .jsonl rows.")
    ap.add_argument("--out", default="variant_comparison.md")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    paths = sorted(metrics_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(
            f"No .jsonl rows in {metrics_dir}/. Runs write them via "
            f"--variant-metrics-path; check the agents actually finished."
        )
    rows = load_rows(paths)
    counts: dict[str, int] = {}
    for row in rows:
        vid = row.get("variant_id")
        if vid:
            counts[vid] = counts.get(vid, 0) + 1

    Path(args.out).write_text(render(rows, counts))
    print(f"[collect] {len(rows)} rows from {len(paths)} file(s) -> {args.out}")
    for vid in VARIANT_ORDER:
        print(f"  {vid}: {counts.get(vid, 0)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
