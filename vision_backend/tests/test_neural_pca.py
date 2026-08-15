"""Tests for the neural-PCA explainability method (pc_align/neural_pca.py)."""

import numpy as np
import pytest
import torch

from vision_backend.pc_align.neural_pca import (
    build_gallery_entry,
    compute_psi_batch,
    fit_class_pca,
    load_gallery,
    project_onto_components,
    save_gallery,
)
from vision_backend.training.builders import build_simmim_segmentation_model


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = {
        "model_kind": "simmim",
        "in_channels": 1,
        "global_base_grid": 4,
        "window_size": 8,
        "decoder_channels": 16,
        "num_classes": 3,
    }
    return build_simmim_segmentation_model(config).eval()


def test_compute_psi_batch_shape(tiny_model):
    x = torch.randn(3, 1, 256, 256)
    psi = compute_psi_batch(tiny_model, 1, x)
    assert psi.shape == (3, 16)  # decoder_channels=16
    assert torch.isfinite(psi).all()


def test_fit_class_pca_shapes_and_orthonormality():
    torch.manual_seed(0)
    psi = torch.randn(30, 10)
    pca = fit_class_pca(psi, class_id=2, n_components=4)

    assert pca.class_id == 2
    assert pca.eigvecs.shape == (4, 10)
    assert pca.eigvals.shape == (4,)
    assert pca.num_samples == 30
    # SVD right-singular vectors are orthonormal.
    gram = pca.eigvecs @ pca.eigvecs.T
    assert torch.allclose(gram, torch.eye(4), atol=1e-4)


def test_fit_class_pca_caps_components_to_available_rank():
    psi = torch.randn(3, 10)  # only 3 samples -> rank <= 3
    pca = fit_class_pca(psi, class_id=0, n_components=8)
    assert pca.eigvecs.shape[0] <= 3


def test_project_onto_components_shape():
    torch.manual_seed(1)
    psi = torch.randn(20, 6)
    pca = fit_class_pca(psi, class_id=0, n_components=3)
    scores = project_onto_components(psi, pca)
    assert scores.shape == (20, 3)


def test_project_onto_components_ranks_synthetic_direction_correctly():
    """A component built to align with a known injected direction should rank
    samples with the largest coefficient along that direction highest."""
    torch.manual_seed(2)
    direction = torch.zeros(5)
    direction[0] = 1.0
    coeffs = torch.tensor([0.0, 5.0, -5.0, 10.0, -10.0])
    psi = coeffs[:, None] * direction[None, :] + 0.01 * torch.randn(5, 5)

    pca = fit_class_pca(psi, class_id=0, n_components=1)
    scores = project_onto_components(psi, pca)[:, 0]

    # The two largest-magnitude-coefficient samples should be the top-2 by |score|.
    top2 = torch.topk(scores.abs(), k=2).indices.tolist()
    assert set(top2) == {3, 4}


def test_build_gallery_entry_ranks_and_limits_top_k():
    torch.manual_seed(3)
    psi = torch.randn(10, 4)
    pca = fit_class_pca(psi, class_id=0, n_components=2)
    thumbnails = [np.full((8, 8), i, dtype=np.uint8) for i in range(10)]
    source_ids = [f"crop_{i}" for i in range(10)]

    entry = build_gallery_entry(pca, psi, thumbnails, source_ids, top_k=3)

    assert set(entry.keys()) == {0, 1}
    for component_idx, items in entry.items():
        assert len(items) == 3
        scores = [item.score for item in items]
        assert scores == sorted(scores, reverse=True)
        assert [item.rank for item in items] == [1, 2, 3]
        # Each item's thumbnail should be traceable back to its source_id via the
        # same index (thumbnail value == index used to build it).
        for item in items:
            idx = int(item.source_id.split("_")[1])
            assert item.thumbnail[0, 0] == idx


def test_build_gallery_entry_mismatched_lengths_raises():
    psi = torch.randn(5, 4)
    pca = fit_class_pca(psi, class_id=0, n_components=1)
    with pytest.raises(ValueError, match="matching lengths"):
        build_gallery_entry(pca, psi, thumbnails=[np.zeros((4, 4), dtype=np.uint8)] * 3, source_ids=["a"] * 5)


def test_save_load_gallery_roundtrip(tmp_path):
    torch.manual_seed(4)
    psi = torch.randn(8, 4)
    pca = fit_class_pca(psi, class_id=0, n_components=2)
    thumbnails = [np.random.randint(0, 255, (8, 8), dtype=np.uint8) for _ in range(8)]
    source_ids = [f"s{i}" for i in range(8)]
    entry = build_gallery_entry(pca, psi, thumbnails, source_ids, top_k=2)
    gallery = {0: entry}

    path = tmp_path / "gallery.pt"
    save_gallery(gallery, path)
    loaded = load_gallery(path)

    assert set(loaded.keys()) == {0}
    assert loaded[0][0][0].thumbnail.shape == (8, 8)
    assert loaded[0][0][0].rank == 1


def test_load_gallery_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gallery(tmp_path / "nope.pt")
