"""Per-run measurements for the encoder-variant comparison.

The variant matrix asks what a narrower encoder (and, per the calibration
decision, a correspondingly narrower decoder) costs in accuracy and buys in
size/speed. Accuracy comes from the sweep metric; this module supplies the
cost side -- parameter counts, training step time, inference throughput and
peak memory -- so every run emits the same row and the variants can be compared
without re-deriving numbers by hand afterwards.

All of it is cheap and side-effect free: throughput is measured on synthetic
tensors after training, not by touching the dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


def parameter_breakdown(model) -> dict[str, int]:
    """Encoder / decoder / total parameter counts.

    Reported separately because they behave differently: the encoder scales with
    the width knob, the decoder with `decoder_channels`, and the totals alone
    would hide which half a variant actually changed.
    """
    out = {"params_total": count_parameters(model)}
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    if encoder is not None:
        out["params_encoder"] = count_parameters(encoder)
    if decoder is not None:
        out["params_decoder"] = count_parameters(decoder)
    return out


def peak_memory_gb(torch_module, device) -> Optional[float]:
    """Peak allocated memory in GB, or None where the backend can't report it."""
    try:
        if device.type == "cuda":
            return float(torch_module.cuda.max_memory_allocated(device)) / 1e9
        if device.type == "mps":
            return float(torch_module.mps.current_allocated_memory()) / 1e9
    except Exception:
        return None
    return None


def reset_peak_memory(torch_module, device) -> None:
    try:
        if device.type == "cuda":
            torch_module.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def measure_inference_throughput(
    torch_module,
    model,
    device,
    *,
    input_size: int = 512,
    batch_size: int = 2,
    needs_context: bool = False,
    warmup: int = 2,
    iters: int = 5,
) -> dict[str, float]:
    """Forward-only throughput on synthetic input, in images/second.

    Synthetic rather than dataset-driven so the number reflects the model alone
    -- no loader, no disk -- which is what makes it comparable across variants.
    """
    was_training = model.training
    model.eval()
    local = torch_module.randn(batch_size, 1, input_size, input_size, device=device)
    context = (
        torch_module.randn(batch_size, 1, input_size, input_size, device=device)
        if needs_context
        else None
    )
    args = (local, context) if needs_context else (local,)

    def _sync():
        if device.type == "cuda":
            torch_module.cuda.synchronize()
        elif device.type == "mps":
            torch_module.mps.synchronize()

    with torch_module.no_grad():
        for _ in range(warmup):
            model(*args)
        _sync()
        start = time.time()
        for _ in range(iters):
            model(*args)
        _sync()
        elapsed = time.time() - start

    model.train(was_training)
    images = batch_size * iters
    return {
        "infer_images_per_s": images / elapsed if elapsed > 0 else 0.0,
        "infer_ms_per_image": (elapsed / images) * 1000.0 if images else 0.0,
    }


def write_variant_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one run's measurements as a JSON line.

    JSONL rather than CSV so a run can add a field without invalidating earlier
    rows, and so a partially-finished sweep is still readable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
