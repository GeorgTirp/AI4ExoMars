"""§8.7 — data-prep golden checks: crop count, rejection, range, split stability."""

from pathlib import Path

import numpy as np

from vision_backend.hirise_patchloader import ArrayImageReader
from vision_backend import prep_crops as pc


def _synthetic_observation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 210, size=(2048, 2048)).astype(np.uint8)  # 4 tiles
    img[0:1024, 0:1024] = 0       # tile (0,0): all invalid -> rejected
    img[0:1024, 1024:2048] = 128  # tile (0,1): constant -> low-variance rejected
    return img  # tiles (1,0), (1,1): textured -> kept


def test_crop_count_and_rejection(tmp_path: Path):
    reader = ArrayImageReader(Path("synthetic.jp2"), _synthetic_observation(), "array")
    records, stats = pc.process_observation(
        reader,
        "OBS_0001",
        source_rel="synthetic.jp2",
        output_dir=tmp_path,
        crop_size=1024,
        max_invalid_fraction=0.01,
        min_std=0.02,
        gsd=0.25,
        dry_run=False,
        overwrite=True,
    )
    assert stats == {"candidates": 4, "kept": 2, "rejected_invalid": 1, "rejected_lowvar": 1}
    assert len(records) == 2


def test_stored_crop_dequantizes_to_unit_range(tmp_path: Path):
    reader = ArrayImageReader(Path("synthetic.jp2"), _synthetic_observation(1), "array")
    records, _ = pc.process_observation(
        reader, "OBS_0001", source_rel="x.jp2", output_dir=tmp_path,
        crop_size=1024, max_invalid_fraction=0.01, min_std=0.02, gsd=0.5,
        dry_run=False, overwrite=True,
    )
    arr = np.load(tmp_path / records[0].crop_path)
    assert arr.dtype == np.uint8
    deq = arr / 127.5 - 1.0
    assert deq.min() >= -1.0 - 1e-6 and deq.max() <= 1.0 + 1e-6


def test_split_is_by_observation_and_stable():
    obs = [f"O{i:03d}" for i in range(100)]
    s1 = pc.split_by_observation(obs, val_fraction=0.02, min_val=20, seed=42)
    s2 = pc.split_by_observation(obs, val_fraction=0.02, min_val=20, seed=42)
    assert s1 == s2  # stable across reruns
    assert len(s1["val"]) == 20  # min_val enforced (2% of 100 = 2 < 20)
    assert set(s1["train"]).isdisjoint(s1["val"])
    assert set(s1["train"]) | set(s1["val"]) == set(obs)


def test_normalize_crop_rejects_constant():
    flat = np.full((1024, 1024), 100, dtype=np.uint8)
    u8, std = pc.normalize_crop(flat, np.zeros_like(flat, dtype=bool))
    assert u8 is None and std == 0.0
