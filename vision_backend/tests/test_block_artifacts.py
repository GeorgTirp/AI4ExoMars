"""Uncertainty / explainability files of the Stage-4 block classifier:
block_features, the last-layer Laplace (GGN against autograd, predictive
limits, save/load), the per-block MC Fisher, block neural-PCA bases and
vector Mahalanobis scoring, and scripts/fit_block_artifacts.py end to end on a
tiny model (block level and campaign posterior; the NOAH-H pixel fits need the
real mosaic)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vision_backend.laplace_prior import DiagonalLaplacePrior, mc_fisher_diagonal_blocks, save_prior
from vision_backend.model.block_head import BlockClassifier
from vision_backend.pc_align.neural_pca import (compute_block_psi, fit_class_pca, load_gallery, load_gallery_bases,
                                                save_gallery)
from vision_backend.training.builders import build_simmim_segmentation_model
from vision_backend.uncertainty.block_laplace import (LastLayerLaplace, last_layer_ggn, load_last_layer_laplace,
                                                      save_last_layer_laplace, tune_prior_precision)
from vision_backend.uncertainty.malahanobis import (fit_class_gaussians, mahalanobis_distance,
                                                   mahalanobis_distance_map)

REPO = Path(__file__).resolve().parents[2]
CFG = {"model_kind": "simmim", "in_channels": 1, "global_base_grid": 8, "window_size": 8,
       "decoder_channels": 16, "num_classes": 5}


def _model(seed=0):
    torch.manual_seed(seed)
    seg = build_simmim_segmentation_model(dict(CFG)).eval()
    model = BlockClassifier(seg, list(range(5)) + [None, None]).eval()
    with torch.no_grad():
        model.fc.weight.normal_(0, 0.3)
        model.fc.bias.normal_(0, 0.3)
    return model


def test_block_features_reproduce_the_forward_pass():
    model = _model()
    x = torch.randn(2, 1, 256, 256)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[0, :, :, 128:] = False
    with torch.no_grad():
        phi, offset = model.block_features(x, valid)
        assert phi.shape == (2, 16 + 5) and offset.shape == (2, 7)
        assert torch.equal(model(x, valid), offset + model.fc(phi))


def test_last_layer_ggn_matches_autograd():
    torch.manual_seed(0)
    n, d, k = 6, 4, 3
    phi, off = torch.randn(n, d, dtype=torch.float64), torch.randn(n, k, dtype=torch.float64)
    w = torch.randn(k, d + 1, dtype=torch.float64)
    ggn = torch.zeros(k * (d + 1), k * (d + 1), dtype=torch.float64)
    for i in range(n):
        def f(wflat):
            wm = wflat.view(k, d + 1)
            return off[i] + wm[:, :d] @ phi[i] + wm[:, d]
        jac = torch.autograd.functional.jacobian(f, w.flatten())          # [K, P]
        p = torch.softmax(f(w.flatten()), 0)
        ggn += jac.T @ (torch.diag(p) - torch.outer(p, p)) @ jac
    logits = off + phi @ w[:, :d].T + w[:, d]
    torch.testing.assert_close(last_layer_ggn(phi, logits, chunk=4), ggn)


def _laplace(n=40, d=5, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n, d, generator=g)
    w = torch.randn(k, d + 1, generator=g)
    logits = phi @ w[:, :d].T + w[:, d]
    return LastLayerLaplace.fit(phi, logits, w, prior_precision=1.0), phi, logits


def test_laplace_predictive_limits_and_centering():
    la, phi, logits = _laplace()
    cov = la.logit_covariance(phi)
    torch.testing.assert_close(cov.sum(2), torch.zeros(len(phi), la.num_classes), atol=1e-5, rtol=0)
    assert (cov.diagonal(dim1=1, dim2=2) >= -1e-6).all()
    # an infinitely sharp posterior is the MAP
    sharp = la.predictive(logits, phi, method="probit", prior_precision=1e9)
    torch.testing.assert_close(sharp["probs"], torch.softmax(logits, 1), atol=1e-4, rtol=0)
    mc = la.predictive(logits, phi, method="mc", n_samples=200, prior_precision=1e9,
                       generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(mc["probs"], torch.softmax(logits, 1), atol=1e-4, rtol=0)
    assert float(mc["mutual_information"].max()) < 1e-4
    # a broad posterior is less confident and has epistemic uncertainty
    broad = la.predictive(logits, phi, method="mc", n_samples=500, prior_precision=1e-2,
                          generator=torch.Generator().manual_seed(0))
    map_entropy = -(torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum(1)
    assert float(broad["entropy"].mean()) > float(map_entropy.mean())
    assert float(broad["mutual_information"].mean()) > 1e-3
    torch.testing.assert_close(broad["entropy"], broad["aleatoric"] + broad["mutual_information"], atol=1e-5, rtol=0)


def test_laplace_tuning_save_load_and_model_check(tmp_path):
    la, phi, logits = _laplace()
    labels = torch.distributions.Categorical(logits=logits).sample()
    lam, table = tune_prior_precision(la, logits, phi, labels, grid=torch.logspace(-2, 4, 7))
    assert lam in [t[0] for t in table] and all(np.isfinite(t[1]) for t in table)
    la.prior_precision = lam
    save_last_layer_laplace(la, tmp_path / "la.pt")
    back = load_last_layer_laplace(tmp_path / "la.pt")
    assert back.prior_precision == lam
    torch.testing.assert_close(back.logit_covariance(phi), la.logit_covariance(phi))
    fc = torch.nn.Linear(5, 4)
    with torch.no_grad():
        fc.weight.copy_(la.weight[:, :5])
        fc.bias.copy_(la.weight[:, 5])
    back.check_model(fc)
    with torch.no_grad():
        fc.bias.add_(1.0)
    try:
        back.check_model(fc)
        raise AssertionError("check_model accepted a different head")
    except ValueError:
        pass


def test_block_fisher_covers_every_parameter():
    model = _model()
    batches = [{"image": torch.randn(1, 1, 256, 256), "valid": torch.ones(1, 1, 256, 256, dtype=torch.bool)}
               for _ in range(3)]
    fisher, n = mc_fisher_diagonal_blocks(model, batches, device=torch.device("cpu"))
    assert n == 3 and set(fisher) == {k for k, _ in model.named_parameters()}
    assert all((f >= 0).all() for f in fisher.values())
    assert float(fisher["fc.weight"].sum()) > 0 and float(fisher["seg.decoder.head.weight"].sum()) > 0


def test_block_psi_bases_only_gallery_and_vector_mahalanobis(tmp_path):
    torch.manual_seed(0)
    w = torch.randn(3, 6)
    phi = torch.randn(30, 6)
    torch.testing.assert_close(compute_block_psi(w, phi, 2), phi * w[2])
    bases = {2: fit_class_pca(compute_block_psi(w, phi, 2), class_id=2, n_components=4)}
    save_gallery({}, tmp_path / "b.npca.pt", bases=bases)
    assert load_gallery(tmp_path / "b.npca.pt") == {}
    assert set(load_gallery_bases(tmp_path / "b.npca.pt")) == {2}

    labels = torch.randint(0, 3, (30,))
    stats = fit_class_gaussians(phi, labels)
    via_map = mahalanobis_distance_map(phi.T.reshape(1, 6, 5, 6), stats).reshape(-1)
    torch.testing.assert_close(mahalanobis_distance(phi, stats), via_map)


def test_fit_block_artifacts_script_end_to_end(tmp_path):
    """Tiny checkpoint + manifest of 256-px blocks; block-level files and the campaign posterior."""
    model = _model()
    run = tmp_path / "run"
    run.mkdir()
    names = [f"DC{i}" for i in range(5)] + ["NewA", "NewB"]
    seg_index = list(range(5)) + [None, None]
    (tmp_path / "block_classes.json").write_text(json.dumps(
        {"classes": [{"name": n, "seg_index": s} for n, s in zip(names, seg_index)]}))
    dc = tmp_path / "dc.json"
    dc.write_text(json.dumps({"classes": [{"id": i + 1, "name": f"DC{i}"} for i in range(5)]}))
    rng = np.random.default_rng(0)
    rows = []
    (tmp_path / "crops").mkdir()
    for i in range(36):
        split = ("train", "train", "val", "test")[i % 4]
        img = f"crops/b{i}.png"
        Image.fromarray(rng.integers(1, 255, (256, 256)).astype(np.uint8)).save(tmp_path / img)
        rows.append({"image": img, "block_id": f"b{i}", "size": 256, "label": names[i % 7], "split": split,
                     "obs_id": "OBS", "cell": f"OBS:{i // 4}:0"})
    with (tmp_path / "manifest.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    prior_path = tmp_path / "noahh_prior.pt"
    save_prior(prior_path, {k: p.detach() for k, p in model.seg.named_parameters()},
               {k: torch.ones_like(p) for k, p in model.seg.named_parameters()})
    torch.save({"block_state": model.state_dict(), "seg_model_config": dict(CFG),
                "classes": {"names": names, "seg_index": list(range(5)) + [None, None]},
                "head_config": {"attn_hidden": 128, "dropout": 0.1, "tile_size": 256},
                "source_checkpoint": "unused", "epoch": 1,
                "args": {"manifest": str(tmp_path / "manifest.csv"), "block_classes": str(tmp_path / "block_classes.json"),
                         "dc_classes_json": str(dc), "laplace_prior": str(prior_path), "laplace_tau": 2.0}},
               run / "best.pt")
    post = tmp_path / "posterior.pt"
    cmd = [sys.executable, str(REPO / "scripts" / "fit_block_artifacts.py"), str(run / "best.pt"), "--skip-pixel",
           "--posterior-out", str(post), "--num-workers", "0", "--mc-samples", "50", "--min-npca-samples", "2",
           "--limit-blocks", "1000"]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    summary = json.loads((run / "artifacts.json").read_text())
    assert summary["block_feature_check"]["test_accuracy_from_features"] >= 0
    assert set(summary["block_level"]["last_layer_laplace"]["test"]) == {"map", "map_temperature", "laplace_probit",
                                                                         "laplace_mc"}
    la = load_last_layer_laplace(run / "best.laplace.pt")
    la.check_model(model.fc)
    assert la.n_fit == 18 and la.eigvecs.shape == (7 * 22, 7 * 22)
    train_counts = np.bincount([i % 7 for i in range(36) if i % 4 in (0, 1)], minlength=7)
    assert set(load_gallery_bases(run / "best.npca.pt")) == {k for k in range(7) if train_counts[k] >= 2}

    # the campaign posterior: tau * F_NOAH-H + N * F_expert over every parameter, usable as the next prior
    p = torch.load(post, map_location="cpu", weights_only=False)
    assert p["n_expert"] == 18 and p["tau_noahh"] == 2.0
    assert set(p["fisher"]) == {k for k, _ in model.named_parameters()}
    for k, lam in p["fisher"].items():
        expected = 18 * p["fisher_expert_mean"][k] + (2.0 if k.startswith("seg.") else 0.0)
        torch.testing.assert_close(lam, expected)
    prior = DiagonalLaplacePrior(post, model, tau=1.0)
    assert float(prior.penalty()) == 0.0

    # refit from the saved features without the imagery
    for f in ("best.laplace.pt", "best.uncertainty.pt", "best.npca.pt"):
        (run / f).unlink()
    subprocess.run(cmd + ["--from-features"], cwd=REPO, check=True, capture_output=True)
    assert all((run / f).is_file() for f in ("best.laplace.pt", "best.uncertainty.pt", "best.npca.pt"))
    torch.testing.assert_close(load_last_layer_laplace(run / "best.laplace.pt").eigvals, la.eigvals)
