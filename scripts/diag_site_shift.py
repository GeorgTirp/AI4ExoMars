#!/usr/bin/env python3
"""Why does the NOAH-H segmenter collapse on a new observation? Image statistics.

Compares the Oxia DRG crops the segmenter was trained on (crop cache, central
512) with the expert blocks as the model sees them (prep_expert_block_manifest
output). Brightness/contrast are matched by construction, so this looks at what
is left: gray-level count, texture sharpness (radially averaged power spectrum
of the contrast-normalised crop, gradient energy), noise (Immerkaer 1996), and
the illumination direction (skewness of the directional derivatives: slopes
facing the sun are lit, cast shadows give sharp dark edges, so the derivative
distribution is skewed along the sun azimuth). Writes a JSON summary and a
montage of random crops from both.

    python scripts/diag_site_shift.py --n 150
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

DER = Path("data/2022-02-08_ABarrett_OU_HiRISE_NOAH-H_Mosaic/derived")
BANDS = ((0.0, 0.03), (0.03, 0.1), (0.1, 0.25), (0.25, 0.5))  # cycles / px (0.25 m px)


def conv3(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] - 2, a.shape[1] - 2))
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * a[dy:dy + a.shape[0] - 2, dx:dx + a.shape[1] - 2]
    return out


def skew(v: np.ndarray) -> float:
    v = v - v.mean()
    return float((v ** 3).mean() / max(v.std() ** 3, 1e-12))


def crop_stats(u8: np.ndarray) -> dict | None:
    valid = u8 > 0
    if valid.mean() < 0.98:
        return None  # stats on partially nodata crops would mix in the fill
    a = u8.astype(np.float64)
    std = a.std()
    if std < 1e-6:
        return None
    z = (a - a.mean()) / std
    f = np.fft.fftshift(np.abs(np.fft.fft2(z * np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1])))) ** 2)
    n = z.shape[0]
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n)), np.fft.fftshift(np.fft.fftfreq(n)), indexing="ij")
    r = np.hypot(fy, fx)
    total = f[(r > 0) & (r <= 0.5)].sum()
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    gx, gy = conv3(z, sobel) / 8, conv3(z, sobel.T) / 8
    lap = conv3(a, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float))
    noise = np.sqrt(np.pi / 2) * np.abs(lap).mean() / 6  # Immerkaer 1996, in gray levels
    angles = np.deg2rad(np.arange(0, 360, 15))
    skews = [skew(np.cos(t) * gx + np.sin(t) * gy) for t in angles]
    k = int(np.argmax(skews))
    return {
        "mean": a.mean(), "std": std, "levels": int(np.unique(u8).size),
        **{f"psd_{lo}-{hi}": float(f[(r > lo) & (r <= hi)].sum() / total) for lo, hi in BANDS},
        "grad_rms": float(np.sqrt((gx ** 2 + gy ** 2).mean())),
        "noise_over_std": noise / std,
        "max_skew": skews[k], "max_skew_azimuth_deg": float(np.rad2deg(angles[k])),
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        v = np.array([r[key] for r in rows], dtype=float)
        if key == "max_skew_azimuth_deg":
            ang = np.deg2rad(v)
            mean = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
            out[key] = {"circular_mean": round(float(mean), 1),
                        "resultant_length": round(float(np.hypot(np.sin(ang).mean(), np.cos(ang).mean())), 3)}
        else:
            q = np.percentile(v, [25, 50, 75])
            out[key] = {"p25": round(float(q[0]), 4), "median": round(float(q[1]), 4), "p75": round(float(q[2]), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crop-cache-dir", default=str(DER / "seg_crop_cache_full"))
    ap.add_argument("--expert-manifest", default="data/BlockLabels_prepared/manifest.csv")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="results/block_finetune/site_shift.json")
    ap.add_argument("--montage", default="results/block_finetune/site_shift_montage.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = json.loads((Path(args.crop_cache_dir) / "meta.json").read_text())
    images = np.load(Path(args.crop_cache_dir) / "images.npy", mmap_mode="r")
    m, S = int(meta["jitter_margin"]), int(meta["crop_size"])
    oxia_crops = []
    for i in np.sort(rng.choice(images.shape[0], size=4 * args.n, replace=False)):
        c = np.asarray(images[i, m:m + S, m:m + S])
        if (c > 0).mean() >= 0.98:
            oxia_crops.append(c)
        if len(oxia_crops) == args.n:
            break

    manifest = Path(args.expert_manifest)
    with manifest.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    pick = rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)
    expert_crops = [np.array(Image.open(manifest.parent / rows[i]["image"])) for i in pick]

    result = {}
    for name, crops in (("oxia_training", oxia_crops), ("expert_" + rows[0]["obs_id"], expert_crops)):
        stats = [s for s in map(crop_stats, crops) if s is not None]
        result[name] = {"n": len(stats), **summarize(stats)}

    names = list(result)
    print(f"{'statistic (median [p25, p75])':<26}" + "".join(f"{n:>34}" for n in names))
    for key in result[names[0]]:
        if key == "n":
            print(f"{'n crops':<26}" + "".join(f"{result[n]['n']:>34}" for n in names))
            continue
        cells = []
        for n in names:
            s = result[n][key]
            cells.append(f"{s['circular_mean']:.0f} deg (R={s['resultant_length']:.2f})" if "circular_mean" in s
                         else f"{s['median']:.4g} [{s['p25']:.4g}, {s['p75']:.4g}]")
        print(f"{key:<26}" + "".join(f"{c:>34}" for c in cells))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    tiles = [oxia_crops[i] for i in range(4)] + [expert_crops[i] for i in range(4)]
    w = 256
    canvas = Image.new("L", (4 * w + 3 * 8, 2 * w + 8), 255)
    for k, t in enumerate(tiles):
        canvas.paste(Image.fromarray(t).resize((w, w), Image.BILINEAR), ((k % 4) * (w + 8), (k // 4) * (w + 8)))
    canvas.save(args.montage)
    print(f"\nwrote {args.out_json} and {args.montage} (top row Oxia training crops, bottom row expert blocks)")


if __name__ == "__main__":
    main()
