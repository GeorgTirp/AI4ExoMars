"""Neural PCA: class-wise PCA in feature space, for "what did the model learn" explainability.

Ported from the method in
``Maritan-Terrain-Sematic-Segmentation/src/martian_terrain_segmentation/explainability.py``
(itself following lecture-slide notation), adapted to AI4ExoMars's segmentation
models via ``model/features.py`` instead of a model-specific ``get_cam_layer()``.

Method, per class k:

1. phi(x)  = GAP(features feeding the classifier head)          -- [F]
2. w_k     = classifier weight vector for class k                -- [F]
3. psi_k(x) = w_k (elementwise*) phi(x)                          -- [F] ("class-conditioned" embedding)
4. PCA over {psi_k(x_i)} for samples x_i containing class k       -- top components = the class's
   dominant directions of variation in *class-relevant* feature space
5. Projecting samples onto each component and ranking by score finds the images that most
   strongly activate that direction -- "what does the model see when it's confident about this
   class, along its L most distinct axes of variation".

This module is pure math/model-hooking (testable against the real, even
untrained, model classes). Turning a dataset into psi vectors + ranked
thumbnail galleries is dataset-shape-specific and lives in the offline
``fit_neural_pca.py`` script, which calls into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from vision_backend.model.features import extract_pooled_features, get_classifier_weight_vector
except ModuleNotFoundError:
    from model.features import extract_pooled_features, get_classifier_weight_vector

__all__ = [
    "NeuralPCAResult",
    "GalleryThumbnail",
    "NeuralPCAGallery",
    "compute_psi_batch",
    "fit_class_pca",
    "project_onto_components",
    "build_gallery_entry",
    "save_gallery",
    "load_gallery",
]


@dataclass
class NeuralPCAResult:
    """Fitted PCA over one class's psi_k(x) embeddings."""

    class_id: int
    mean_psi: torch.Tensor  # [D]
    eigvecs: torch.Tensor  # [L, D], top-L principal directions
    eigvals: torch.Tensor  # [L]
    num_samples: int


@dataclass
class GalleryThumbnail:
    """One ranked example for a (class, component) pair, portable for GUI display."""

    rank: int
    score: float
    thumbnail: np.ndarray  # uint8 [H, W] crop, small enough to embed in the artifact
    source_id: str  # provenance string (e.g. "OBS123_2048_1024")


# class_id -> component_idx (0-based) -> ranked thumbnails (best first)
NeuralPCAGallery = dict[int, dict[int, list[GalleryThumbnail]]]


def compute_psi_batch(model: torch.nn.Module, class_id: int, *inputs: torch.Tensor) -> torch.Tensor:
    """psi_k(x) for a batch: elementwise w_k * phi(x). Shape [B, F].

    `*inputs` is forwarded to the model as in `model.features.extract_pixel_features`
    (one tensor for the single-branch model, local+context for the context model).
    """
    w_k = get_classifier_weight_vector(model, class_id)  # [F]
    phi = extract_pooled_features(model, *inputs)  # [B, F]
    return phi * w_k


def fit_class_pca(psi: torch.Tensor, class_id: int, *, n_components: int = 4) -> NeuralPCAResult:
    """PCA over a class's collected psi_k(x_i) vectors.

    Parameters
    ----------
    psi : torch.Tensor
        [N, D] psi_k(x_i) vectors, one row per sample known to contain class_id.
    class_id : int
    n_components : int
        Number of orthogonal directions to keep (the "4 orthogonal features").
    """
    if psi.ndim != 2:
        raise ValueError(f"psi must be [N, D], got shape {tuple(psi.shape)}")
    psi = psi.detach().float()
    mean_psi = psi.mean(dim=0, keepdim=True)
    centered = psi - mean_psi

    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, vh.shape[0])
    eigvecs = vh[:n_components]  # [L, D]
    eigvals = (s[:n_components] ** 2) / max(1, psi.shape[0] - 1)

    return NeuralPCAResult(
        class_id=class_id,
        mean_psi=mean_psi[0],
        eigvecs=eigvecs,
        eigvals=eigvals,
        num_samples=psi.shape[0],
    )


def project_onto_components(psi: torch.Tensor, pca: NeuralPCAResult) -> torch.Tensor:
    """Per-sample activation score on each component. Shape [N, L].

    Scaled by each eigenvector's own coefficient sum (`eigvecs.sum(dim=1)`),
    matching the reference implementation's convention -- this is a fixed
    per-component scale factor, so it doesn't change the *ranking* of samples
    within a component, only the score's magnitude/sign.
    """
    diff = psi.detach().float() - pca.mean_psi  # [N, D]
    proj = diff @ pca.eigvecs.T  # [N, L]
    ones_dot_v = pca.eigvecs.sum(dim=1)  # [L]
    return proj * ones_dot_v


def build_gallery_entry(
    pca: NeuralPCAResult,
    psi: torch.Tensor,
    thumbnails: Sequence[np.ndarray],
    source_ids: Sequence[str],
    *,
    top_k: int = 6,
) -> dict[int, list[GalleryThumbnail]]:
    """Rank `thumbnails` on each of pca's components, keeping the top_k per component.

    `psi`, `thumbnails`, and `source_ids` must be the same length and row-aligned
    (row i of psi <-> thumbnails[i] <-> source_ids[i]).
    """
    if not (len(thumbnails) == len(source_ids) == psi.shape[0]):
        raise ValueError("psi, thumbnails, and source_ids must have matching lengths")

    scores = project_onto_components(psi, pca)  # [N, L]
    result: dict[int, list[GalleryThumbnail]] = {}
    for component_idx in range(scores.shape[1]):
        comp_scores = scores[:, component_idx]
        k = min(top_k, comp_scores.numel())
        top_vals, top_idx = torch.topk(comp_scores, k=k)
        result[component_idx] = [
            GalleryThumbnail(
                rank=rank + 1,
                score=float(top_vals[rank]),
                thumbnail=thumbnails[int(top_idx[rank])],
                source_id=source_ids[int(top_idx[rank])],
            )
            for rank in range(k)
        ]
    return result


def save_gallery(gallery: NeuralPCAGallery, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gallery, path)


def load_gallery(path: str | Path) -> NeuralPCAGallery:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No fitted neural-PCA gallery at {path}")
    # weights_only=False: first-party artifact (dict of dataclasses + numpy arrays).
    return torch.load(path, map_location="cpu", weights_only=False)
