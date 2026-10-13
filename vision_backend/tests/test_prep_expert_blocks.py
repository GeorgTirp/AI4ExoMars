"""prep_expert_block_manifest.py on a synthetic two-observation export (512 and
1024 blocks): model_index class mapping, kept splits, whole vs quartered 1024
training blocks, spatial CV folds, and the CV split used for training."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_backend.block_dataset import load_block_manifest
from vision_backend.train_stage4_block_finetune import cv_splits

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "prep_expert_block_manifest.py"
CLASSES = [{"id": i, "name": n, "target_index": i, "model_index": i}
           for i, n in enumerate(["Bedrock: Smooth", "Bedrock: Rugged", "Craters", "Dunes"])]


def _export(root: Path, obs: str, size: int, grid: int, rng) -> Path:
    d = root / obs / "dataset-1"
    (d / "crops").mkdir(parents=True)
    rows, integrity = [], {}
    for gy in range(grid):
        for gx in range(grid):
            x, y = gx * size, gy * size
            name = f"{obs}_{x}_{y}.png"
            Image.fromarray(rng.integers(300, 800, size=(size, size)).astype(np.uint16)).save(d / "crops" / name)
            integrity[f"crops/{name}"] = hashlib.sha256((d / "crops" / name).read_bytes()).hexdigest()
            cid = int(rng.integers(0, 4))
            rows.append({"block_id": f"{obs}_{x}_{y}", "obs_id": obs, "x_px": x, "y_px": y, "class_id": cid,
                         "target_index": cid, "class_name": CLASSES[cid]["name"], "w_px": size, "h_px": size,
                         "crop_filename": name, "dtype": "uint16", "edit_count": 1})
    with (d / "labels.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (d / "classes.json").write_text(json.dumps({"classes": CLASSES}))
    (d / "integrity.json").write_text(json.dumps({"algorithm": "sha256", "files": integrity}))
    (d / "dataset.json").write_text(json.dumps({
        "format": "mars-block-crops", "target_kind": "whole_block_majority_class", "label_origin": "labeling_session",
        "block_size": size, "observation_id": obs,
        "preprocessing": {"method": "global_percentile_uint8", "bounds": [330.0, 799.0]}}))
    return d


def _run(*args, cwd):
    subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=cwd, check=True, capture_output=True)


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def exports(tmp_path):
    rng = np.random.default_rng(0)
    dc = tmp_path / "dc.json"
    dc.write_text(json.dumps({"classes": [{"id": 1, "name": "Smooth bedrock"}, {"id": 2, "name": "Rugged bedrock"}]}))
    a = _export(tmp_path / "labels", "OBS_A", 512, 12, rng)
    b = _export(tmp_path / "labels", "OBS_B", 1024, 8, rng)
    return tmp_path, dc, a, b


def test_kept_split_whole_and_quartered_blocks_and_cv(exports):
    tmp, dc, a, b = exports
    common = ["--dc-classes-json", dc, "--cell-px", 2048, "--trials", 20]
    _run("--exports", a, "--out-dir", tmp / "v1", *common, cwd=REPO)
    v1 = {r["block_id"]: r for r in _rows(tmp / "v1" / "manifest.csv")}
    for variant, extra in (("v2", []), ("v2q", ["--quarter-train"])):
        _run("--exports", a, b, "--block-sizes", 512, 1024, "--keep-splits-from", tmp / "v1" / "manifest.csv",
             "--cv-folds", 3, "--out-dir", tmp / variant, *extra, *common, cwd=REPO)
    v2, v2q = _rows(tmp / "v2" / "manifest.csv"), _rows(tmp / "v2q" / "manifest.csv")

    # the first observation keeps its split and cell; the second gets train/val/test of its own
    for r in v2:
        if r["obs_id"] == "OBS_A":
            assert (r["split"], r["cell"]) == (v1[r["block_id"]]["split"], v1[r["block_id"]]["cell"])
    assert {r["split"] for r in v2 if r["obs_id"] == "OBS_B"} >= {"train", "val", "test"}
    # model_index < 2 -> DC id; Craters / Dunes are new classes in the fixed list
    names = [c["name"] for c in json.loads((tmp / "v2" / "block_classes.json").read_text())["classes"]]
    assert names == ["Smooth bedrock", "Rugged bedrock", "Craters", "Dunes"]

    # whole 1024 blocks in v2; in v2q only the TRAIN ones are quartered, with the parent's label and fold
    assert {int(r["size"]) for r in v2 if r["obs_id"] == "OBS_B"} == {1024}
    quarters = [r for r in v2q if r["obs_id"] == "OBS_B" and r["split"] == "train"]
    parents = [r for r in v2 if r["obs_id"] == "OBS_B" and r["split"] == "train"]
    assert len(quarters) == 4 * len(parents) and all(int(r["size"]) == 512 for r in quarters)
    by_parent = {r["block_id"]: r for r in parents}
    for q in quarters:
        p = by_parent[q["parent_block_id"]]
        assert (q["label"], q["fold"], q["cv_buffer"], q["cell"]) == (p["label"], p["fold"], p["cv_buffer"], p["cell"])
        img = np.array(Image.open(tmp / "v2q" / q["image"]))
        assert img.shape == (512, 512)
    assert all(int(r["size"]) == 1024 for r in v2q if r["obs_id"] == "OBS_B" and r["split"] != "train")

    # CV: every train/val block has a fold, whole cells share it, borders are exactly the neighbours' folds
    tv = [r for r in v2 if r["split"] in ("train", "val")]
    assert all(r["fold"] != "" for r in tv) and {r["fold"] for r in tv} == {"0", "1", "2"}
    assert all(r["fold"] == "" for r in v2 if r["split"] in ("test", "buffer"))
    cell_folds = {}
    for r in tv:
        cell_folds.setdefault(r["cell"], set()).add(r["fold"])
    assert all(len(f) == 1 for f in cell_folds.values())
    pos = {(r["obs_id"], int(r["x_px"]), int(r["y_px"])): r for r in tv}
    for r in tv:
        s = int(r["size"])
        near = {pos[k]["fold"] for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy)
                and (k := (r["obs_id"], int(r["x_px"]) + dx * s, int(r["y_px"]) + dy * s)) in pos}
        assert set(r["cv_buffer"].split(";")) - {""} == near - {r["fold"]}

    # the training-side CV split: held-out fold validates, bordering blocks are dropped, test unused
    records, _ = load_block_manifest(tmp / "v2" / "manifest.csv", dc, tmp / "v2" / "block_classes.json")
    s = cv_splits(records, 1)
    assert s["test"] == [] and all(r.fold == 1 for r in s["val"])
    assert all(r.fold != 1 and 1 not in r.cv_buffer for r in s["train"])
    assert all(r.split in ("train", "val") for r in s["train"] + s["val"])
