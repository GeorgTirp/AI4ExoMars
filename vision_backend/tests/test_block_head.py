"""Stage-4 block classifier: zero-shot equivalence with the segmenter's
dominant-class rule, nodata handling, the new-class path, manifest class
resolution, the block dataset and checkpoint round trip."""

import csv
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vision_backend.block_dataset import BlockDataset, load_block_manifest, resolve_block_classes
from vision_backend.model.block_head import BlockClassifier, GatedAttentionPool2d
from vision_backend.training.builders import build_simmim_segmentation_model
from vision_backend.train_stage4_block_finetune import load_block_classifier, seg_model_config_from, summarize

CFG = {"model_kind": "simmim", "in_channels": 1, "global_base_grid": 8, "window_size": 8,
       "decoder_channels": 16, "num_classes": 5}


def _model(num_new=2, seed=0):
    torch.manual_seed(seed)
    seg = build_simmim_segmentation_model(dict(CFG)).eval()
    return BlockClassifier(seg, list(range(5)) + [None] * num_new).eval()


def _pixel_log_probs(model, x):
    with torch.no_grad():
        return F.log_softmax(model.seg.decoder.head(model.seg.decode_features(x)), dim=1)


def test_init_is_the_segmenters_dominant_class_rule():
    model = _model()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        logits = model(x)
    pbar = _pixel_log_probs(model, x).exp().mean(dim=(2, 3))
    torch.testing.assert_close(logits[:, :5], pbar.log(), atol=1e-5, rtol=0)
    torch.testing.assert_close(logits[:, 5:], torch.full((2, 2), math.log(1 / 7)))
    assert torch.equal(logits[:, :5].argmax(1), pbar.argmax(1))


def test_nodata_is_excluded_from_the_prior():
    model = _model()
    x = torch.randn(1, 1, 256, 256)
    valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    valid[..., :, 128:] = False  # right half is nodata (aligned to the 1/4 grid)
    with torch.no_grad():
        logits = model(x, valid)
    log_p = _pixel_log_probs(model, x)
    expected = log_p[..., :, : log_p.shape[-1] // 2].exp().mean(dim=(2, 3)).log()
    torch.testing.assert_close(logits[:, :5], expected, atol=1e-5, rtol=0)


def test_attention_pool_ignores_zero_weight_positions():
    torch.manual_seed(0)
    pool = GatedAttentionPool2d(8, 4)
    x = torch.randn(1, 8, 6, 6)
    w = torch.ones(1, 1, 6, 6)
    w[..., 3:] = 0
    x2 = x.clone()
    x2[..., 3:] = 100 * torch.randn(1, 8, 6, 3)
    torch.testing.assert_close(pool(x, w), pool(x2, w), atol=1e-3, rtol=0)


def test_new_class_is_learned_by_the_head():
    model = _model()
    for p in model.seg.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.head_parameters(), lr=1e-2)
    x = torch.randn(4, 1, 256, 256)
    y = torch.full((4,), 6)  # the second new class
    model.train()
    model.seg.eval()
    losses = []
    for _ in range(25):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]
    model.eval()
    with torch.no_grad():
        assert (model(x).argmax(1) == 6).all()
    assert all(p.grad is None for p in model.seg.parameters())


def _dc_json(tmp_path):
    classes = [{"id": i + 1, "name": n} for i, n in enumerate(
        ["Smooth + Featureless", "Textured bedrock", "Continous + Simple form large ripples"])]
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"classes": classes}))
    return path


def test_manifest_class_resolution(tmp_path):
    dc = _dc_json(tmp_path)
    rows = [("smooth  +  featureless", "train"), ("Continuous + Simple form large ripples", "train"),
            ("2", "val"), ("Dune field", "val"), ("Crater", "train")]
    manifest = tmp_path / "blocks.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["col", "row", "label", "split"])
        for i, (label, split) in enumerate(rows):
            w.writerow([i * 512, 0, label, split])
    records, classes = load_block_manifest(manifest, dc)
    assert classes.names[3:] == ["Crater", "Dune field"]
    assert classes.seg_index == [0, 1, 2, None, None] and classes.num_new == 2
    assert [r.label for r in records] == [0, 2, 1, 4, 3]
    assert [r.split for r in records] == ["train", "train", "val", "val", "train"]
    # a typo becomes a new class -- which is what --expect-new-classes guards against
    assert resolve_block_classes(["Textured bedrok"], dc).num_new == 1


def test_block_dataset_reads_mosaic_windows_and_tiles(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    rng = np.random.default_rng(0)
    mosaic = rng.integers(1, 256, size=(1200, 1200), dtype=np.uint8)
    mosaic[:, :100] = 0  # nodata strip
    profile = {"driver": "GTiff", "width": 1200, "height": 1200, "count": 1, "dtype": "uint8"}
    with rasterio.open(tmp_path / "mosaic.tif", "w", **profile) as ds:
        ds.write(mosaic, 1)
    with rasterio.open(tmp_path / "tile.tif", "w", **{**profile, "width": 512, "height": 512}) as ds:
        ds.write(mosaic[:512, 600:1112], 1)

    from vision_backend.block_dataset import BlockRecord
    recs = [BlockRecord(label=1, split="val", col=0, row=300),
            BlockRecord(label=2, split="val", image=str(tmp_path / "tile.tif"))]
    ds = BlockDataset(recs, imagery_path=tmp_path / "mosaic.tif", jitter=32)
    item = ds[0]
    assert item["image"].shape == (1, 512, 512) and item["label"] == 1
    np.testing.assert_allclose(item["image"][0].numpy(), mosaic[300:812, :512] / 127.5 - 1.0, atol=1e-6)
    assert not item["valid"][0, :, :100].any() and item["valid"][0, :, 100:].all()
    np.testing.assert_allclose(ds[1]["image"][0].numpy(), mosaic[:512, 600:1112] / 127.5 - 1.0, atol=1e-6)

    aug = BlockDataset(recs, imagery_path=tmp_path / "mosaic.tif", jitter=32, augment=True)
    for _ in range(5):
        it = aug[0]
        assert it["image"].shape == (1, 512, 512)
        assert (it["image"][~it["valid"]] == -1.0).all()


def test_checkpoint_round_trip(tmp_path):
    model = _model()
    with torch.no_grad():
        model.fc.weight.normal_()
    path = tmp_path / "best.pt"
    torch.save({"block_state": model.state_dict(),
                "seg_model_config": seg_model_config_from(model.seg, {"model": dict(CFG)}),
                "classes": {"names": [str(i) for i in range(7)], "seg_index": model.seg_index},
                "head_config": {"attn_hidden": 128, "dropout": 0.1}}, path)
    loaded, _ = load_block_classifier(path)
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


def test_stage4_script_end_to_end(tmp_path, monkeypatch):
    """main(): zero-shot eval, probe epoch, unfreeze with the routed Muon/NAdamW
    optimizer and a Laplace prior, full fine-tuning epoch, best checkpoint, test split."""
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("muon")
    import sys
    from vision_backend import train_stage4_block_finetune as stage4

    torch.manual_seed(0)
    seg = build_simmim_segmentation_model(dict(CFG, num_classes=3))
    torch.save({"model_state": seg.state_dict(), "config": {"model": dict(CFG)}}, tmp_path / "seg.pt")
    from vision_backend.laplace_prior import save_prior
    save_prior(tmp_path / "prior.pt", dict(seg.named_parameters()),
               {n: torch.ones_like(p) for n, p in seg.named_parameters()}, source_checkpoint="seg.pt")
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
    before = {n: p.detach().clone() for n, p in seg.named_parameters()}

    monkeypatch.setattr(sys, "argv", [
        "stage4", "--seg-checkpoint", str(tmp_path / "seg.pt"), "--manifest", str(tmp_path / "blocks.csv"),
        "--imagery-path", str(tmp_path / "mosaic.tif"), "--dc-classes-json", str(_dc_json(tmp_path)),
        "--out-dir", str(tmp_path / "out"), "--epochs", "2", "--probe-epochs", "1", "--batch-size", "2",
        "--num-workers", "0", "--expect-new-classes", "1", "--jitter", "16", "--block-size", "256",
        "--laplace-prior", str(tmp_path / "prior.pt"), "--laplace-tau", "100"])
    stage4.main()

    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    with (tmp_path / "out" / "history.csv").open(newline="") as fh:
        penalties = [float(r["laplace_penalty"]) for r in csv.DictReader(fh)]
    assert penalties[0] == 0.0 and penalties[1] > 0.0  # probe epoch: no prior; full fine-tuning: active
    assert {"zero_shot_val", "best_val", "test", "best_epoch"} <= set(metrics)
    model, ckpt = load_block_classifier(tmp_path / "out" / "best.pt")
    assert ckpt["classes"]["names"][3:] == ["Crater"]
    changed = [n for n, p in model.seg.named_parameters() if not torch.equal(p, before[n])]
    if ckpt["epoch"] == 2:  # the full fine-tuning epoch was the best one: the segmenter moved
        assert any("encoder.s3" in n for n in changed) and any("decoder" in n for n in changed)
    else:  # probe epoch: the segmenter is untouched
        assert not changed


def test_large_blocks_are_tiled_at_the_backbone_input_size():
    model = _model()
    model.tile_size = 256  # the HybridEncoder needs multiples of 256
    x = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        stitched = model.decode_features(x)
        tiles = [[model.seg.decode_features(x[..., r:r + 256, c:c + 256]) for c in (0, 256)] for r in (0, 256)]
    expected = torch.cat([torch.cat(row, dim=3) for row in tiles], dim=2)
    torch.testing.assert_close(stitched, expected)
    with torch.no_grad():
        assert model(x).shape == (1, 7)


def test_size_grouped_batches():
    from vision_backend.block_dataset import SizeGroupedBatchSampler
    sides = [512] * 5 + [1024] * 3
    for shuffle in (False, True):
        sampler = SizeGroupedBatchSampler(sides, 4, shuffle=shuffle)
        batches = list(sampler)
        assert len(batches) == len(sampler) == 2 + 3
        assert sorted(i for b in batches for i in b) == list(range(8))
        assert all(len({sides[i] for i in b}) == 1 for b in batches)
        assert all(len(b) == 1 for b in batches if sides[b[0]] == 1024)


def test_fixed_class_list(tmp_path):
    from vision_backend.block_dataset import load_block_classes
    fixed = [{"name": "Smooth + Featureless", "seg_index": 0}, {"name": "Textured bedrock", "seg_index": 1},
             {"name": "Craters", "seg_index": None}, {"name": "Dunes", "seg_index": None}]
    (tmp_path / "block_classes.json").write_text(json.dumps({"classes": fixed}))
    with (tmp_path / "m.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "split"])
        w.writerow(["a.png", "2", "train"])
        w.writerow(["b.png", "Craters", "val"])
    records, classes = load_block_manifest(tmp_path / "m.csv", _dc_json(tmp_path), tmp_path / "block_classes.json")
    assert classes.names == ["Smooth + Featureless", "Textured bedrock", "Craters", "Dunes"]
    assert classes.num_new == 2 and [r.label for r in records] == [1, 2]
    (tmp_path / "bad.json").write_text(json.dumps({"classes": fixed[2:] + fixed[:2]}))
    with pytest.raises(ValueError):
        load_block_classes(tmp_path / "bad.json")


def test_summarize():
    from vision_backend.block_dataset import BlockClasses
    classes = BlockClasses(names=["a", "b", "c"], seg_index=[0, 1, None])
    labels = np.array([0, 0, 1, 2])
    logits = np.log(np.array([[.9, .05, .05], [.2, .7, .1], [.1, .8, .1], [.1, .1, .8]]))
    m = summarize(logits, labels, classes)
    assert m["accuracy"] == 0.75
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert m["balanced_accuracy_noahh"] == pytest.approx(0.75) and m["balanced_accuracy_new"] == 1.0
    assert m["confusion"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert m["nll"] == pytest.approx(-np.mean(np.log([.9, .2, .8, .8])))
