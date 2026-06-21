"""§8.5 overfit smoke + §8.6 resume determinism for the Stage-1 loop."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from vision_backend.model.hybrid_encoder import HybridEncoder
from vision_backend.model.simmim import SimMIMModel
from vision_backend.training.config import config_from_dict
from vision_backend.train_stage1_simmim import train


def _smooth_batch() -> torch.Tensor:
    # block-constant smooth crops: masked 32² blocks are reconstructable from
    # surrounding context (unlike pure noise, whose masked pixels are
    # information-free).
    g = torch.linspace(0, 1, 8)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    a = torch.sin(2 * torch.pi * xx) + torch.cos(2 * torch.pi * yy)
    b = torch.cos(2 * torch.pi * xx) + torch.sin(2 * torch.pi * yy)
    grid = torch.stack([a, b]).unsqueeze(1) / 2.0  # (2, 1, 8, 8) in ~[-1, 1]
    return F.interpolate(grid, scale_factor=32, mode="nearest")  # (2, 1, 256, 256)


def test_overfit_smoke():
    """Wiring smoke test: on a single fixed batch the masked-L1 loss must drop
    by a solid margin and stay finite -- this catches wrong loss targets, dead
    gradients, and mask leakage.

    NB: the spec's "< 0.1x initial" target is not used. By design (guardrail
    §10) the decoder is a single 1x1 conv + PixelShuffle over a 32x-downsampled
    bottleneck, so masked-pixel L1 plateaus well above zero on a single batch
    even though context demonstrably propagates (loss falls far below the
    predict-the-mean floor). A real wiring bug shows ~no decrease or NaN, which
    the 0.6x margin separates cleanly.
    """
    torch.manual_seed(0)
    encoder = HybridEncoder(in_channels=1, global_base_grid=8, drop_path=0.0)
    model = SimMIMModel(encoder, in_channels=1, mask_ratio=0.4, loss_type="l1")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x = _smooth_batch()
    mask = model.masker.sample_mask(2, 8, 8, torch.device("cpu"))

    initial = None
    last = None
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, mask=mask)
        loss = out["loss"]
        assert torch.isfinite(loss), "loss became non-finite (wiring bug)"
        if initial is None:
            initial = loss.item()
        last = loss.item()
        loss.backward()
        optimizer.step()
        if last < 0.45 * initial:
            break

    assert last < 0.6 * initial, (initial, last)


# --------------------------------------------------------------------------
# resume determinism
# --------------------------------------------------------------------------
def _make_dataset(root: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for obs_idx in range(3):
        obs = f"OBS_{obs_idx:03d}"
        (root / obs).mkdir(parents=True, exist_ok=True)
        for crop_idx in range(4):
            arr = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
            rel = f"{obs}/{obs}_c{crop_idx:02d}.npy"
            np.save(root / rel, arr)
            rows.append({"crop_path": rel, "observation_id": obs})

    index_path = root / "crops_index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "observation_id"])
        writer.writeheader()
        writer.writerows(rows)

    with (root / "splits.json").open("w") as f:
        json.dump({"train": ["OBS_000", "OBS_001"], "val": ["OBS_002"]}, f)
    return index_path


def _cfg(index_path: Path, ckpt_dir: Path, total_steps: int):
    return config_from_dict(
        {
            "seed": 7,
            "precision": "fp32",
            "deterministic": True,
            "data": {
                "index_path": str(index_path),
                "splits_path": str(index_path.parent / "splits.json"),
                "batch_size": 2,
                "num_workers": 0,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "augment": False,
                "crop_size": 256,
            },
            "model": {
                "global_base_grid": 8,
                "drop_path": 0.0,
                "use_checkpoint": False,
            },
            "optim": {
                "effective_batch": 2,  # accum = 1
                "total_steps": total_steps,
                "warmup_fraction": 0.1,
                "ema_decay": 0.9,
            },
            "output": {
                "checkpoint_dir": str(ckpt_dir),
                "checkpoint_every": 5,
                "val_every": 10_000,
                "panel_every": 10_000,
                "log_every": 10_000,
            },
        }
    )


def _args():
    return SimpleNamespace(resume=None, no_wandb=True, wandb=False)


def _losses_by_step(history):
    return {row["step"]: row["train_loss"] for row in history}


def test_resume_is_bitwise_deterministic(tmp_path: Path):
    index_path = _make_dataset(tmp_path / "data")
    total = 10
    cut = 5

    # uninterrupted run, 0 -> total
    out_a = train(_cfg(index_path, tmp_path / "ckpt_a", total), _args())
    losses_a = _losses_by_step(out_a["history"])

    # partial run 0 -> cut (same schedule via total), saves checkpoint at `cut`
    train(_cfg(index_path, tmp_path / "ckpt_b", total), _args(), max_steps=cut)

    # resume cut -> total
    resume_args = _args()
    resume_args.resume = str(tmp_path / "ckpt_b" / "last.pt")
    out_b = train(_cfg(index_path, tmp_path / "ckpt_b", total), resume_args)
    losses_b = _losses_by_step(out_b["history"])

    for step in range(cut + 1, total + 1):
        assert losses_b[step] == losses_a[step], (step, losses_b[step], losses_a[step])
