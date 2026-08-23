"""DC (14-class) -> IG (5-class) hierarchy for the NOAH-H taxonomy.

Stage-3 performance-levers brief, F1 (hierarchical IG aux head): the 14
Descriptive Classes (DC) roll up to 5 Interpretive Groups (IG). No IG legend
mosaic is present locally (only the DC mosaic is -- see classes_DC.json), so
there is no tile-legend CSV to decode a real classes_IG.json from the way
noahh_alignment/common.py:write_classes_json produced classes_DC.json.
DEFAULT_DC_TO_IG below is transcribed directly from the brief's taxonomy table
instead, and data/.../derived/classes_IG.json + dc_to_ig.json mirror it in a
loadable form for --dc-to-ig-path, so a real legend can replace this file
later without a code change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

IG_CLASSES = [
    {"id": 1, "name": "Non-bedrock"},
    {"id": 2, "name": "Bedrock"},
    {"id": 3, "name": "Large ripples"},
    {"id": 4, "name": "Small ripples"},
    {"id": 5, "name": "Other cover"},
]

# 1-based DC id (classes_DC.json) -> 1-based IG id (classes_IG.json).
DEFAULT_DC_TO_IG: dict[int, int] = {
    1: 1, 2: 1, 3: 1,        # Non-bedrock: Smooth+Featureless, Smooth+Lineated, Textured non-bedrock
    4: 2, 5: 2, 6: 2, 7: 2,  # Bedrock: Smooth / Textured / Rugged / Fractured bedrock
    8: 3, 9: 3, 10: 3,       # Large ripples: Continuous+Simple, Isolated+Simple, Rectilinear
    11: 4, 12: 4, 13: 4,     # Small ripples: Continuous small, [non-]bedrock-substrate non-continuous small
    14: 5,                   # Other cover: Boulder fields
}


def load_dc_to_ig_mapping(path: Optional[str | Path]) -> dict[int, int]:
    """Load a {dc_id: ig_id} mapping from a dc_to_ig.json (see the default at
    data/.../derived/dc_to_ig.json), or return DEFAULT_DC_TO_IG if path is None."""
    if path is None:
        return dict(DEFAULT_DC_TO_IG)
    raw = json.loads(Path(path).read_text())["dc_to_ig"]
    return {int(k): int(v) for k, v in raw.items()}


def num_ig_classes(mapping: Optional[dict[int, int]] = None) -> int:
    mapping = mapping or DEFAULT_DC_TO_IG
    return len(set(mapping.values()))


def build_dc_to_ig_tensor(torch_module, num_classes_dc: int, mapping: Optional[dict[int, int]] = None):
    """0-based LongTensor[num_classes_dc]: DC class index -> IG class index.

    `mapping` (default DEFAULT_DC_TO_IG) is 1-based {dc_id: ig_id}, matching
    classes_DC.json's ids; seg_dataset.py remaps DC labels 1..C -> 0..C-1 for
    training, so this validates every id 1..num_classes_dc is mapped and IG
    ids form a dense 1..N range, then returns 0-based indices ready to apply
    directly to those remapped DC labels/logits.
    """
    mapping = mapping or DEFAULT_DC_TO_IG
    missing = [dc_id for dc_id in range(1, num_classes_dc + 1) if dc_id not in mapping]
    if missing:
        raise ValueError(f"dc_to_ig mapping is missing DC id(s): {missing}")
    ig_ids = sorted(set(mapping[dc_id] for dc_id in range(1, num_classes_dc + 1)))
    if ig_ids != list(range(1, len(ig_ids) + 1)):
        raise ValueError(f"IG ids must be a dense 1..N range starting at 1, got {ig_ids}")
    dc_to_ig = [mapping[dc_id] - 1 for dc_id in range(1, num_classes_dc + 1)]
    return torch_module.tensor(dc_to_ig, dtype=torch_module.long)


def map_dc_labels_to_ig(torch_module, dc_labels, dc_to_ig, ignore_index: int):
    """Map a DC-label tensor (any shape, values in [0, num_classes_dc) or
    `ignore_index`) to IG labels of the same shape, preserving ignore_index.

    Uses the focal_loss-style dummy-safe-index trick (training/utils.py):
    ignored positions are substituted with a safe index before the gather so
    out-of-range `ignore_index` values (e.g. 255) never index into dc_to_ig,
    then re-masked back to ignore_index afterwards.
    """
    ignore_mask = dc_labels == ignore_index
    safe_dc = dc_labels.masked_fill(ignore_mask, 0)
    ig_labels = dc_to_ig.to(dc_labels.device)[safe_dc]
    return ig_labels.masked_fill(ignore_mask, ignore_index)


def aggregate_dc_class_counts_to_ig(dc_counts: dict[int, int], dc_to_ig) -> dict[int, int]:
    """Sum 0-based DC pixel counts into 0-based IG pixel counts (for IG-head
    class weights/priors) without a second pass over the raster labels."""
    dc_to_ig_list = dc_to_ig.tolist()
    ig_counts: dict[int, int] = {}
    for dc_idx, count in dc_counts.items():
        ig_idx = dc_to_ig_list[dc_idx]
        ig_counts[ig_idx] = ig_counts.get(ig_idx, 0) + count
    return ig_counts
