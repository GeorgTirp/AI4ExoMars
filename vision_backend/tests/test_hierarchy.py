"""Tests for training/hierarchy.py (F1: DC -> IG mapping)."""

import json

import pytest
import torch

from vision_backend.training.hierarchy import (
    DEFAULT_DC_TO_IG,
    aggregate_dc_class_counts_to_ig,
    build_dc_to_ig_tensor,
    load_dc_to_ig_mapping,
    map_dc_labels_to_ig,
    num_ig_classes,
)

IGNORE_INDEX = 255


def test_default_mapping_covers_every_dc_id_1_to_14():
    assert set(DEFAULT_DC_TO_IG.keys()) == set(range(1, 15))


def test_num_ig_classes_is_five():
    assert num_ig_classes() == 5


def test_build_dc_to_ig_tensor_matches_expected_groups():
    dc_to_ig = build_dc_to_ig_tensor(torch, 14)
    assert dc_to_ig.dtype == torch.long
    assert dc_to_ig.shape == (14,)
    # 0-based DC index -> 0-based IG index (id - 1).
    expected = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    assert dc_to_ig.tolist() == expected


def test_build_dc_to_ig_tensor_raises_on_missing_dc_id():
    incomplete = {k: v for k, v in DEFAULT_DC_TO_IG.items() if k != 7}
    with pytest.raises(ValueError, match="missing DC id"):
        build_dc_to_ig_tensor(torch, 14, mapping=incomplete)


def test_build_dc_to_ig_tensor_raises_on_non_dense_ig_ids():
    bad = dict(DEFAULT_DC_TO_IG)
    bad[14] = 9  # skips ids 6..8, no longer a dense 1..N range
    with pytest.raises(ValueError, match="dense"):
        build_dc_to_ig_tensor(torch, 14, mapping=bad)


def test_map_dc_labels_to_ig_preserves_ignore_index():
    dc_to_ig = build_dc_to_ig_tensor(torch, 14)
    dc_labels = torch.tensor([[0, 3, 13], [IGNORE_INDEX, 7, IGNORE_INDEX]])
    ig_labels = map_dc_labels_to_ig(torch, dc_labels, dc_to_ig, ignore_index=IGNORE_INDEX)
    assert ig_labels.shape == dc_labels.shape
    assert ig_labels[0, 0].item() == 0  # DC 1 ("Smooth+Featureless") -> IG 1 ("Non-bedrock")
    assert ig_labels[0, 1].item() == 1  # DC 4 ("Smooth bedrock") -> IG 2 ("Bedrock")
    assert ig_labels[0, 2].item() == 4  # DC 14 ("Boulder fields") -> IG 5 ("Other cover")
    assert ig_labels[1, 0].item() == IGNORE_INDEX
    assert ig_labels[1, 1].item() == 2  # DC 8 -> IG 3 ("Large ripples")
    assert ig_labels[1, 2].item() == IGNORE_INDEX


def test_map_dc_labels_to_ig_all_ignored_does_not_crash():
    dc_to_ig = build_dc_to_ig_tensor(torch, 14)
    dc_labels = torch.full((2, 3), IGNORE_INDEX, dtype=torch.long)
    ig_labels = map_dc_labels_to_ig(torch, dc_labels, dc_to_ig, ignore_index=IGNORE_INDEX)
    assert torch.all(ig_labels == IGNORE_INDEX)


def test_aggregate_dc_class_counts_to_ig_sums_correctly():
    dc_to_ig = build_dc_to_ig_tensor(torch, 14)
    # 0-based DC counts: classes 0,1,2 (all -> IG 0) and class 13 (-> IG 4).
    dc_counts = {0: 100, 1: 50, 2: 25, 13: 10}
    ig_counts = aggregate_dc_class_counts_to_ig(dc_counts, dc_to_ig)
    assert ig_counts == {0: 175, 4: 10}


def test_load_dc_to_ig_mapping_from_file_matches_default(tmp_path):
    path = tmp_path / "dc_to_ig.json"
    path.write_text(json.dumps({"dc_to_ig": {str(k): v for k, v in DEFAULT_DC_TO_IG.items()}}))
    loaded = load_dc_to_ig_mapping(str(path))
    assert loaded == DEFAULT_DC_TO_IG


def test_load_dc_to_ig_mapping_none_returns_default():
    assert load_dc_to_ig_mapping(None) == DEFAULT_DC_TO_IG


def test_real_dc_to_ig_json_artifact_is_valid():
    """The checked-in data/.../derived/dc_to_ig.json must load and match
    DEFAULT_DC_TO_IG (it's meant to be the same mapping in loadable form)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "data" / "2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic" / "derived" / "dc_to_ig.json"
    assert path.exists(), f"missing {path}"
    assert load_dc_to_ig_mapping(str(path)) == DEFAULT_DC_TO_IG
