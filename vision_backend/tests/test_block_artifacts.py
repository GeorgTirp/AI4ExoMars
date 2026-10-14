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
from vision_backend.uncertainty.block_laplace import (LastLayerLaplace, dirichlet_from_normcdf, fit_temperature,
                                                      last_layer_ggn, load_last_layer_laplace, normcdf_moments,
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

    # the next campaign: start from this model with its posterior as the prior over every weight,
    # head included -- the prior acts already in the probe epoch, where only the head moves (the
    # unfreeze / full fine-tuning path is test_block_head.py::test_stage4_script_end_to_end)
    from vision_backend import train_stage4_block_finetune as stage4
    torch.save({"model_state": model.seg.state_dict(), "config": {"model": dict(CFG)}}, tmp_path / "seg.pt")
    argv = ["stage4", "--seg-checkpoint", str(tmp_path / "seg.pt"), "--manifest", str(tmp_path / "manifest.csv"),
            "--block-classes", str(tmp_path / "block_classes.json"), "--dc-classes-json", str(dc),
            "--epochs", "1", "--probe-epochs", "1", "--batch-size", "2", "--num-workers", "0",
            "--block-size", "256", "--init-block-checkpoint", str(run / "best.pt")]
    old = sys.argv
    try:
        sys.argv = argv + ["--out-dir", str(tmp_path / "next"), "--laplace-prior", str(post), "--laplace-tau", "1"]
        stage4.main()
    finally:
        sys.argv = old
    first = json.loads((tmp_path / "next" / "metrics.json").read_text())["zero_shot_val"]
    with torch.no_grad():
        x = torch.stack([torch.from_numpy(np.array(Image.open(tmp_path / r["image"]), dtype=np.float32) / 127.5 - 1)
                         for r in rows if r["split"] == "val"])[:, None]
        start = model(x).argmax(1).numpy()
    labels = np.array([names.index(r["label"]) for r in rows if r["split"] == "val"])
    assert abs(first["accuracy"] - float((start == labels).mean())) < 1e-6   # starts exactly at the old model
    with (tmp_path / "next" / "history.csv").open(newline="") as fh:
        penalties = [float(r["laplace_penalty"]) for r in csv.DictReader(fh)]
    assert penalties[0] > 0.0   # the head is under the prior from the first (probe) epoch


# --------------------------------------------------------------------------- normcdf head (E18)
def test_normcdf_head_starts_as_the_dominant_class_rule_and_matches_softmax_at_init():
    from vision_backend.model.block_head import block_log_probs
    torch.manual_seed(0)
    seg = build_simmim_segmentation_model(dict(CFG)).eval()
    soft = BlockClassifier(seg, list(range(5)) + [None, None]).eval()
    norm = BlockClassifier(seg, list(range(5)) + [None, None], link="normcdf").eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        p_soft = block_log_probs(soft(x), "softmax").exp()
        p_norm = block_log_probs(norm(x), "normcdf").exp()
    torch.testing.assert_close(p_norm, p_soft, atol=1e-4, rtol=0)


def test_normcdf_bce_loss_and_fisher():
    from vision_backend.train_stage4_block_finetune import block_loss
    torch.manual_seed(0)
    f = torch.randn(5, 4, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 1])
    w = torch.tensor([1.0, 2.0, 0.5, 1.0])
    p = torch.distributions.Normal(0, 1).cdf(f)
    ref = F.binary_cross_entropy(p, F.one_hot(y, 4).float(), reduction="none").sum(1)
    torch.testing.assert_close(block_loss(f, y, "normcdf", w), (w[y] * ref).sum() / w[y].sum())
    torch.testing.assert_close(block_loss(f, y, "softmax", w), F.cross_entropy(f, y, weight=w))
    # GGN of the per-class Bernoulli(Phi(f)) likelihood against autograd: the Fisher in logit space
    # is E_y[grad grad^T] = diag(phi(f)^2 / (Phi(f) (1 - Phi(f))))
    phi, logits = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64)
    ggn = last_layer_ggn(phi, logits, link="normcdf")
    expected = torch.zeros_like(ggn)
    for n in range(3):
        fn = logits[n].clone().requires_grad_(True)
        pc = torch.distributions.Normal(0, 1).cdf(fn)
        fish = torch.zeros(4, dtype=torch.float64)
        for k in range(4):
            g1, = torch.autograd.grad(torch.log(pc[k]), fn, retain_graph=True)
            g0, = torch.autograd.grad(torch.log(1 - pc[k]), fn, retain_graph=True)
            fish[k] = pc[k].detach() * g1[k] ** 2 + (1 - pc[k].detach()) * g0[k] ** 2
        x = torch.cat([phi[n], torch.ones(1, dtype=torch.float64)])
        expected += torch.kron(torch.diag(fish), torch.outer(x, x))
    torch.testing.assert_close(ggn, expected)


def test_normcdf_closed_form_and_dirichlet_against_monte_carlo():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(6, 5, generator=g, dtype=torch.float64) * 1.5
    var = torch.rand(6, 5, generator=g, dtype=torch.float64) * 3
    m1, m2 = normcdf_moments(mu, var)
    f = mu[None] + var.sqrt()[None] * torch.randn(200000, 6, 5, generator=g, dtype=torch.float64)
    q = torch.distributions.Normal(0, 1).cdf(f)
    torch.testing.assert_close(m1, q.mean(0), atol=4e-3, rtol=0)
    torch.testing.assert_close(m2, (q ** 2).mean(0), atol=4e-3, rtol=0)
    gamma = dirichlet_from_normcdf(mu, var)
    torch.testing.assert_close(gamma / gamma.sum(1, keepdim=True), m1 / m1.sum(1, keepdim=True))
    assert (gamma > 0).all()
    # the Laplace "probit" predictive of a normcdf head: per class exact, normalised by a ratio of
    # expectations -- close to the Monte Carlo predictive for a posterior of realistic width
    la, phi, logits = _laplace()
    la.link = "normcdf"
    cf = la.predictive(logits, phi, method="probit", prior_precision=20.0)["probs"]
    mc = la.predictive(logits, phi, method="mc", prior_precision=20.0, n_samples=20000,
                       generator=torch.Generator().manual_seed(1))["probs"]
    torch.testing.assert_close(cf, mc, atol=0.03, rtol=0)
    broad = la.predictive(logits, phi, method="mc", prior_precision=0.5, n_samples=200,
                          generator=torch.Generator().manual_seed(1))["probs"]
    assert torch.isfinite(broad).all()  # every Phi underflowing must not give 0/0
    d = la.predictive(logits, phi, method="dirichlet", prior_precision=20.0)
    torch.testing.assert_close(d["probs"], cf, atol=1e-5, rtol=0)
    assert (d["mutual_information"] >= 0).all()


def test_temperature_fit_matches_the_evaluation_script():
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_block_classifier import fit_temperature as fit_t_eval
    g = torch.Generator().manual_seed(0)
    logits = 3 * torch.randn(200, 6, generator=g)
    labels = torch.randint(0, 6, (200,), generator=g)
    assert abs(fit_temperature(logits, labels, "softmax") - fit_t_eval(logits.numpy(), labels.numpy())) < 1e-3


def test_mc_lambda_tuning_uses_common_random_numbers():
    la, phi, logits = _laplace()
    labels = torch.distributions.Categorical(logits=logits).sample()
    a = tune_prior_precision(la, logits, phi, labels, grid=torch.logspace(-1, 2, 4), method="mc", n_samples=64)
    b = tune_prior_precision(la, logits, phi, labels, grid=torch.logspace(-1, 2, 4), method="mc", n_samples=64)
    assert a == b


def test_stage4_trainer_with_normcdf_link(tmp_path, monkeypatch):
    """main() with --link normcdf: zero-shot eval, a probe epoch, checkpoint with the link, test split."""
    rasterio = __import__("pytest").importorskip("rasterio")
    from vision_backend import train_stage4_block_finetune as stage4
    from vision_backend.train_stage4_block_finetune import load_block_classifier

    torch.manual_seed(0)
    seg = build_simmim_segmentation_model(dict(CFG, num_classes=3))
    torch.save({"model_state": seg.state_dict(), "config": {"model": dict(CFG)}}, tmp_path / "seg.pt")
    rng = np.random.default_rng(0)
    profile = {"driver": "GTiff", "width": 1024, "height": 1024, "count": 1, "dtype": "uint8"}
    with rasterio.open(tmp_path / "mosaic.tif", "w", **profile) as ds:
        ds.write(rng.integers(1, 256, size=(1024, 1024), dtype=np.uint8), 1)
    with (tmp_path / "blocks.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["col", "row", "label", "split"])
        for i, (label, split) in enumerate([("1", "train"), ("Crater", "train"), ("2", "train"),
                                            ("Crater", "val"), ("1", "val"), ("3", "test")]):
            w.writerow([(i % 3) * 256, (i // 3) * 256, label, split])
    dc = tmp_path / "dc.json"
    dc.write_text(json.dumps({"classes": [{"id": i + 1, "name": n} for i, n in enumerate(["A", "B", "C"])]}))
    monkeypatch.setattr(sys, "argv", [
        "stage4", "--seg-checkpoint", str(tmp_path / "seg.pt"), "--manifest", str(tmp_path / "blocks.csv"),
        "--imagery-path", str(tmp_path / "mosaic.tif"), "--dc-classes-json", str(dc), "--link", "normcdf",
        "--out-dir", str(tmp_path / "out"), "--epochs", "1", "--probe-epochs", "1", "--batch-size", "2",
        "--num-workers", "0", "--expect-new-classes", "1", "--jitter", "16", "--block-size", "256"])
    stage4.main()
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert np.isfinite(metrics["best_val"]["nll"]) and "test" in metrics
    model, ckpt = load_block_classifier(tmp_path / "out" / "best.pt")
    assert ckpt["head_config"]["link"] == "normcdf" and model.link == "normcdf"


def test_link_cv_fold_evaluation_and_summary(tmp_path):
    """eval_link_cv_fold.py on tiny softmax and normcdf models (no Oxia crops), then summarize_link_cv.py."""
    names = [f"DC{i}" for i in range(5)] + ["NewA", "NewB"]
    seg_index = list(range(5)) + [None, None]
    (tmp_path / "block_classes.json").write_text(json.dumps(
        {"classes": [{"name": n, "seg_index": s} for n, s in zip(names, seg_index)]}))
    dc = tmp_path / "dc.json"
    dc.write_text(json.dumps({"classes": [{"id": i + 1, "name": f"DC{i}"} for i in range(5)]}))
    rng = np.random.default_rng(0)
    (tmp_path / "crops").mkdir()
    rows = []
    for i in range(48):
        img = f"crops/b{i}.png"
        Image.fromarray(rng.integers(1, 255, (256, 256)).astype(np.uint8)).save(tmp_path / img)
        rows.append({"image": img, "block_id": f"b{i:02d}", "size": 256, "label": names[i % 7],
                     "split": "train" if i % 6 else "val", "obs_id": "OBS", "cell": f"OBS:{i // 2}:0",
                     "fold": str((i // 2) % 3), "cv_buffer": ""})
    with (tmp_path / "manifest.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for link in ("softmax", "normcdf"):
        torch.manual_seed(0)
        seg = build_simmim_segmentation_model(dict(CFG)).eval()
        model = BlockClassifier(seg, seg_index, link=link).eval()
        with torch.no_grad():
            model.fc.weight.normal_(0, 0.3)
        (tmp_path / link).mkdir()
        torch.save({"block_state": model.state_dict(), "seg_model_config": dict(CFG),
                    "classes": {"names": names, "seg_index": seg_index},
                    "head_config": {"attn_hidden": 128, "dropout": 0.1, "tile_size": 256, "link": link},
                    "source_checkpoint": "unused", "epoch": 1,
                    "args": {"manifest": str(tmp_path / "manifest.csv"),
                             "block_classes": str(tmp_path / "block_classes.json"), "dc_classes_json": str(dc)}},
                   tmp_path / link / "best.pt")
        for f in range(3):
            subprocess.run([sys.executable, str(REPO / "scripts" / "eval_link_cv_fold.py"), str(tmp_path / link / "best.pt"),
                            "--fold", str(f), "--out", str(tmp_path / "res" / f"{link}.fold{f}"), "--oxia-crops", "0",
                            "--num-workers", "0", "--mc-samples", "64", "--mc-samples-tune", "32"],
                           cwd=REPO, check=True, capture_output=True)
    z = np.load(tmp_path / "res" / "normcdf.fold0.npz")
    assert {"p_laplace_closed", "p_laplace_mc", "s_mutual_information_dirichlet"} <= set(z.files)
    assert np.allclose(z["p_laplace_closed"].sum(1), 1, atol=1e-5)
    subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_link_cv.py"), "--in-dir", str(tmp_path / "res"),
                    "--n-boot", "20"], cwd=REPO, check=True, capture_output=True)
    summary = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert summary["n_heldout_blocks"] == 48
    assert set(summary["paired_difference"]["by_predictive"]) == {"map", "map_temperature", "laplace_closed",
                                                                 "laplace_mc"}
