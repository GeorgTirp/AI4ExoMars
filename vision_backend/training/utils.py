from __future__ import annotations

import csv
import importlib
import random
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def resolve_path(path_str: str, *, root: Optional[Path] = None) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    return path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def select_device(torch_module):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def set_seed(torch_module, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def count_parameters(model, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def save_history(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    ensure_parent_dir(path)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(torch_module, state: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    torch_module.save(state, path)


def load_checkpoint(torch_module, path: Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    return torch_module.load(path, map_location=map_location)


def extract_state_dict(checkpoint: dict[str, Any] | dict[str, Any], *preferred_keys: str) -> dict[str, Any]:
    for key in preferred_keys:
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    return checkpoint


def load_prefixed_state_dict(
    model,
    state_dict: dict[str, Any],
    *,
    prefix: str,
    strict: bool = True,
):
    filtered = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not filtered:
        raise KeyError(f"No state dict entries found with prefix {prefix!r}.")
    return model.load_state_dict(filtered, strict=strict)


def maybe_dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def to_config_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return deepcopy(value)
    raise TypeError(f"Expected dataclass or dict, got {type(value)!r}.")


def set_by_dotted_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in updates.items():
        if "." in key:
            set_by_dotted_path(merged, key, value)
            continue

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value

    return merged


def freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def import_object(qualified_name: str):
    module_name, object_name = qualified_name.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def normalize_loader_bundle(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, dict):
        return bundle

    normalized: dict[str, Any] = {}
    for key in ("train", "val", "test", "train_dataset", "val_dataset", "test_dataset", "num_classes"):
        if hasattr(bundle, key):
            normalized[key] = getattr(bundle, key)
    if not normalized:
        raise TypeError("Loader factory must return a dict or object with train/val loaders.")
    return normalized


def load_loader_bundle(factory_path: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    factory = import_object(factory_path)
    return normalize_loader_bundle(factory(**kwargs))


def parse_segmentation_batch(batch: Any) -> tuple[Any, Any, Optional[Any]]:
    if isinstance(batch, dict):
        local = batch.get("local")
        if local is None:
            local = batch.get("image", batch.get("images"))
        target = batch.get("mask")
        if target is None:
            target = batch.get("target", batch.get("targets"))
        if target is None:
            target = batch.get("label", batch.get("labels"))
        context = batch.get("context")
        if local is None or target is None:
            raise KeyError("Segmentation batch dict must contain local/image and mask/target.")
        return local, target, context

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            local, target = batch
            return local, target, None
        if len(batch) == 3:
            local, context, target = batch
            return local, target, context

    raise TypeError(
        "Unsupported segmentation batch format. Expected dict, (local, target), "
        "or (local, context, target)."
    )


def _compute_segmentation_metrics(
    torch_module,
    logits,
    targets,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float]:
    """Pixel accuracy and mean IoU for ONE batch.

    Retained for callers that genuinely want a per-batch reading. Do NOT build
    an epoch metric by averaging this: IoU is a ratio of pixel counts, so the
    mean of per-batch mIoUs is not the dataset mIoU. Each batch also averages
    over whichever classes happen to appear in it, which silently reweights
    rare classes by how often they co-occur. `run_segmentation_epoch`
    accumulates a global confusion matrix instead -- see
    `_accumulate_confusion` / `_metrics_from_confusion`, and
    scripts/eval_global_miou.py for the same computation offline.

    Built from a single fused confusion matrix rather than a per-class mask
    loop. The loop cost two device syncs per class -- every `.item()` drains
    the CUDA queue before the next kernel can be enqueued -- so with 14 DC
    classes plus the 5-class IG head it spent ~45 syncs on *every* training
    step. At the default batch_size=4 (13,492 steps/epoch on the full NOAH-H
    manifest) that serialization is a large fraction of the step. Same
    arithmetic, computed on-device, synced once at the return.
    """
    with torch_module.no_grad():
        preds = logits.argmax(dim=1)
        valid = targets != ignore_index
        # Keep the histogram's indices in range. `preds` is an argmax so it is
        # always in [0, num_classes); a stray out-of-range label would land in
        # the wrong cell. Such a pixel can never be correct (no prediction
        # matches it), so dropping it here leaves the accuracy numerator
        # unchanged, and `total` below still counts it -- as the loop did.
        in_range = valid & (targets >= 0) & (targets < num_classes)
        t = targets[in_range].reshape(-1)
        p = preds[in_range].reshape(-1)

        conf = torch_module.bincount(
            t * num_classes + p, minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

        intersection = conf.diagonal()                       # per class
        union = conf.sum(0) + conf.sum(1) - intersection
        present = union > 0
        # Classes absent from both prediction and target are skipped, not
        # counted as 0 IoU -- clamp_min(1) only guards those 0/0 cells.
        iou = torch_module.where(
            present,
            intersection.double() / union.clamp_min(1).double(),
            torch_module.zeros_like(union, dtype=torch_module.float64),
        )
        total = valid.sum()
        pixel_acc, miou, total_valid = torch_module.stack(
            [
                intersection.sum().double() / total.clamp_min(1).double(),
                iou.sum() / present.sum().clamp_min(1).double(),
                total.double(),
            ]
        ).tolist()  # the single sync
        if total_valid == 0:
            return {"pixel_acc": 0.0, "miou": 0.0}
        return {"pixel_acc": float(pixel_acc), "miou": float(miou)}


def _accumulate_confusion(
    torch_module,
    conf,
    logits,
    targets,
    num_classes: int,
    ignore_index: int,
):
    """Add one batch's counts into `conf` (num_classes x num_classes, rows =
    target, cols = prediction), entirely on-device.

    No `.item()` and no host transfer, so this costs the training loop nothing
    per step; the single sync happens once per epoch in
    `_metrics_from_confusion`.
    """
    with torch_module.no_grad():
        preds = logits.argmax(dim=1)
        valid = targets != ignore_index
        # argmax is always in [0, num_classes); guard the target side so a
        # stray out-of-range label cannot land in the wrong cell.
        in_range = valid & (targets >= 0) & (targets < num_classes)
        t = targets[in_range].reshape(-1)
        p = preds[in_range].reshape(-1)
        conf += torch_module.bincount(
            t * num_classes + p, minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
    return conf


def _metrics_from_confusion(torch_module, conf) -> dict[str, float]:
    """Global pixel accuracy and mean IoU from one accumulated confusion matrix.

    This is the dataset-level definition -- per-class intersection and union
    summed over every pixel of the split, then averaged over the classes that
    actually occur. It is what scripts/eval_global_miou.py reports and what the
    NOAH-H paper's numbers mean.
    """
    with torch_module.no_grad():
        intersection = conf.diagonal()
        union = conf.sum(0) + conf.sum(1) - intersection
        present = union > 0
        total = conf.sum()
        iou = torch_module.where(
            present,
            intersection.double() / union.clamp_min(1).double(),
            torch_module.zeros_like(union, dtype=torch_module.float64),
        )
        stats = torch_module.stack([
            intersection.sum().double() / total.clamp_min(1).double(),
            iou.sum() / present.sum().clamp_min(1).double(),
            total.double(),
        ]).tolist()
    pixel_acc, miou, total_valid = stats
    if total_valid == 0:
        return {"pixel_acc": 0.0, "miou": 0.0}
    return {"pixel_acc": float(pixel_acc), "miou": float(miou)}


def compute_class_weights(
    torch_module,
    class_pixel_counts: "dict[int, int] | Iterable[int]",
    num_classes: int,
    *,
    scheme: str = "inverse_sqrt",
    clip_max: Optional[float] = 10.0,
) -> "torch_module.Tensor":
    """Per-class weights for CrossEntropyLoss from raw pixel counts, to counter
    class imbalance (e.g. one class at 45% of pixels, several under 1%).

    `class_pixel_counts` may be a {class_index: count} dict (missing indices ->
    0) or a sequence of length num_classes. A class with 0 pixels gets weight 0
    (never contributes to the loss -- there's nothing to learn it from here;
    it also can't destabilize training with an infinite weight).

    scheme:
      - "inverse_sqrt" (default): weight ~ 1/sqrt(freq). Common middle ground --
        upweights rare classes without letting the rarest few dominate the
        gradient the way plain inverse frequency can.
      - "inverse": weight ~ 1/freq. More aggressive.
      - "effective_number": Cui et al. 2019 "Class-Balanced Loss" -- weight ~
        (1-beta)/(1-beta^n). Best default when a few classes have very few
        samples (as here), since plain inverse-frequency weights explode for
        near-zero counts while this saturates smoothly.

    clip_max caps the max/min weight ratio after normalization (default 10x)
    so a handful of near-empty classes don't destabilize training with huge
    gradients; None disables clipping.
    """
    if isinstance(class_pixel_counts, dict):
        counts = [class_pixel_counts.get(c, 0) for c in range(num_classes)]
    else:
        counts = list(class_pixel_counts)
        if len(counts) != num_classes:
            raise ValueError(f"Expected {num_classes} counts, got {len(counts)}")

    counts_t = torch_module.tensor(counts, dtype=torch_module.float64)
    present = counts_t > 0

    if scheme == "inverse_sqrt":
        raw = torch_module.zeros_like(counts_t)
        raw[present] = 1.0 / torch_module.sqrt(counts_t[present])
    elif scheme == "inverse":
        raw = torch_module.zeros_like(counts_t)
        raw[present] = 1.0 / counts_t[present]
    elif scheme == "effective_number":
        beta = 1.0 - 1.0 / float(counts_t[present].min().item())
        raw = torch_module.zeros_like(counts_t)
        raw[present] = (1.0 - beta) / (1.0 - beta ** counts_t[present])
    else:
        raise ValueError(f"Unknown scheme: {scheme!r} (expected inverse_sqrt|inverse|effective_number)")

    if not present.any():
        raise ValueError("All class pixel counts are zero -- cannot compute weights")

    # Normalize so present-class weights average to 1 (keeps the loss's overall
    # magnitude comparable to unweighted CE, only the per-class balance shifts).
    raw = raw * (present.sum() / raw[present].sum())

    if clip_max is not None and present.sum() > 1:
        lo = raw[present].max() / clip_max
        raw = torch_module.where(present, raw.clamp(min=lo.item()), raw)

    return raw.float()


def compute_log_class_priors(
    torch_module,
    class_pixel_counts: "dict[int, int] | Iterable[int]",
    num_classes: int,
    *,
    eps: float = 1e-12,
) -> "torch_module.Tensor":
    """log(pixel frequency) per class, for balanced-softmax / logit-adjusted
    loss (F2). `prior_c = count_c / sum(counts)`, clamped away from 0 so a
    class absent from the training split gets a large-but-finite penalty
    instead of -inf (which would make that logit unusable everywhere, not
    just discouraged). Same `{class_index: count} dict or length-num_classes
    sequence` input convention as `compute_class_weights`.
    """
    if isinstance(class_pixel_counts, dict):
        counts = [class_pixel_counts.get(c, 0) for c in range(num_classes)]
    else:
        counts = list(class_pixel_counts)
        if len(counts) != num_classes:
            raise ValueError(f"Expected {num_classes} counts, got {len(counts)}")

    counts_t = torch_module.tensor(counts, dtype=torch_module.float64)
    total = counts_t.sum()
    if total <= 0:
        raise ValueError("All class pixel counts are zero -- cannot compute priors")
    priors = counts_t / total
    return torch_module.log(priors.clamp_min(eps)).float()


def adjust_logits_for_prior(logits, log_priors, tau: float = 1.0):
    """`logits + tau * log(prior_c)` -- the shared balanced-softmax
    (Ren et al. 2020) / logit-adjusted (Menon et al. 2021) adjustment; the two
    papers differ mainly in how `tau` is motivated/tuned, not the formula
    itself, so `--loss-kind balanced_softmax` and `logit_adjusted` share this
    one implementation. Applied to the logits before CE, so it nudges the
    *decision boundary* toward rare classes without reweighting any pixel's
    gradient magnitude the way class-weighted CE does -- hence F2's rule:
    don't stack this with --class-weight-scheme.
    """
    return logits + tau * log_priors.to(logits.device).view(1, -1, 1, 1)


def focal_loss(
    torch_module,
    logits: "torch_module.Tensor",
    target: "torch_module.Tensor",
    *,
    weight: Optional["torch_module.Tensor"] = None,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> "torch_module.Tensor":
    """Multi-class focal loss (Lin et al. 2017, RetinaNet) for dense (per-pixel)
    targets. Down-weights the loss from examples the model already gets right
    with high confidence -- "easy" pixels, which under class imbalance are
    disproportionately the majority class -- and concentrates gradient on the
    ones it's still getting wrong, rather than letting the model coast on
    already-confident majority-class predictions.

    gamma=0 reduces to plain (optionally weighted) cross-entropy; gamma=2 is
    the paper's default and a reasonable starting point.

    Mirrors torch.nn.CrossEntropyLoss(weight=..., ignore_index=...,
    reduction="mean")'s convention -- mean = sum(weight[t] * loss) /
    sum(weight[t]) over valid pixels -- so switching between "ce" and "focal"
    doesn't shift the loss's overall scale.
    """
    log_probs = torch_module.nn.functional.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    ignore_mask = target == ignore_index
    safe_target = target.masked_fill(ignore_mask, 0)  # dummy, safe index; masked out below

    per_pixel_ce = torch_module.nn.functional.nll_loss(
        log_probs, safe_target, weight=weight, reduction="none"
    )  # [B, H, W]
    pt = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    focal = ((1.0 - pt) ** gamma) * per_pixel_ce
    focal = focal.masked_fill(ignore_mask, 0.0)

    if weight is not None:
        pixel_weight = weight[safe_target].masked_fill(ignore_mask, 0.0)
        denom = pixel_weight.sum().clamp_min(1e-8)
    else:
        denom = (~ignore_mask).sum().clamp_min(1)

    return focal.sum() / denom


def _build_loss_fn(
    torch_module,
    loss_kind: str,
    *,
    weight: Optional["torch_module.Tensor"],
    focal_gamma: float,
    log_priors: Optional["torch_module.Tensor"],
    logit_adjust_tau: float,
    ignore_index: int,
    head_label: str,
):
    """One (logits, target) -> scalar loss closure, shared by the DC and IG
    heads (F1's `total = CE_dc + ig_loss_weight * CE_ig` calls this once per
    head with that head's own weight/priors/ignore_index)."""
    if loss_kind == "ce":
        ce = torch_module.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        return lambda logits, target: ce(logits, target)
    if loss_kind == "focal":
        return lambda logits, target: focal_loss(
            torch_module, logits, target, weight=weight, gamma=focal_gamma, ignore_index=ignore_index,
        )
    if loss_kind in ("balanced_softmax", "logit_adjusted"):
        if log_priors is None:
            raise ValueError(
                f"loss_kind={loss_kind!r} requires {head_label}_log_priors "
                "(training-set class pixel frequencies) -- pass them or use "
                "--loss-kind ce/focal instead."
            )
        ce = torch_module.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)

        def _fn(logits, target):
            adjusted = adjust_logits_for_prior(logits, log_priors, tau=logit_adjust_tau)
            return ce(adjusted, target)

        return _fn
    raise ValueError(
        f"Unknown loss_kind: {loss_kind!r} (expected ce|focal|balanced_softmax|logit_adjusted)"
    )


def run_segmentation_epoch(
    torch_module,
    model,
    dataloader,
    device,
    *,
    num_classes: int,
    ignore_index: int,
    optimizer=None,
    scheduler=None,
    use_amp: bool = False,
    accum_steps: int = 1,
    class_weights: Optional["torch_module.Tensor"] = None,
    loss_kind: str = "ce",
    focal_gamma: float = 2.0,
    logit_adjust_tau: float = 1.0,
    dc_log_priors: Optional["torch_module.Tensor"] = None,
    grad_clip_norm: Optional[float] = None,
    error_on_nonfinite_loss: bool = True,
    progress_desc: Optional[str] = None,
    leave_progress: bool = False,
    # F1: hierarchical IG aux head. All default off/None -- neutral default
    # is the plain single-head DC-only path, byte-identical to before F1.
    dc_to_ig: Optional["torch_module.Tensor"] = None,
    num_classes_ig: Optional[int] = None,
    ig_loss_weight: float = 0.0,
    ig_class_weights: Optional["torch_module.Tensor"] = None,
    ig_log_priors: Optional["torch_module.Tensor"] = None,
    # F5: EMA. None (default) = no EMA tracking, current behavior.
    ema=None,
    ema_source_model=None,
    # Mixed-precision dtype on CUDA. "fp16" (default) is the historical
    # behaviour and needs a GradScaler; "bf16" has fp32's exponent range, so it
    # cannot overflow where fp16 does and needs no scaler.
    amp_dtype: str = "fp16",
    # Lovász-softmax (training/lovasz.py) added to the DC loss with this weight;
    # 0.0 (default) leaves the loss unchanged.
    lovasz_weight: float = 0.0,
) -> dict[str, float]:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    training = optimizer is not None
    accum_steps = max(int(accum_steps), 1)
    model.train(training)

    want_ig = ig_loss_weight > 0
    if want_ig and (dc_to_ig is None or num_classes_ig is None):
        raise ValueError(
            "ig_loss_weight > 0 requires both dc_to_ig and num_classes_ig "
            "(the model must have been built with a matching IG aux head)."
        )

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    if loss_kind in ("balanced_softmax", "logit_adjusted") and weight_tensor is not None:
        # F2: these adjust logits toward the class distribution directly:
        # stacking a second, independent reweighting on top double-counts the
        # imbalance correction and was not what the sweep/tests validate.
        print(
            f"[run_segmentation_epoch] WARNING: loss_kind={loss_kind!r} does not "
            "stack with class_weights -- ignoring the provided class-weight vector."
        )
        weight_tensor = None

    dc_loss_fn = _build_loss_fn(
        torch_module, loss_kind, weight=weight_tensor, focal_gamma=focal_gamma,
        log_priors=dc_log_priors, logit_adjust_tau=logit_adjust_tau,
        ignore_index=ignore_index, head_label="dc",
    )
    if lovasz_weight > 0:
        try:
            from vision_backend.training.lovasz import lovasz_softmax
        except ModuleNotFoundError:
            from training.lovasz import lovasz_softmax
        base_dc_loss_fn = dc_loss_fn

        def dc_loss_fn(logits, target):  # noqa: F811 -- deliberate wrap
            return base_dc_loss_fn(logits, target) + lovasz_weight * lovasz_softmax(
                logits, target, ignore_index=ignore_index
            )

    ig_loss_fn = None
    if want_ig:
        ig_weight_tensor = ig_class_weights.to(device) if ig_class_weights is not None else None
        if loss_kind in ("balanced_softmax", "logit_adjusted") and ig_weight_tensor is not None:
            print(
                f"[run_segmentation_epoch] WARNING: loss_kind={loss_kind!r} does not "
                "stack with ig_class_weights -- ignoring the provided IG class-weight vector."
            )
            ig_weight_tensor = None
        ig_loss_fn = _build_loss_fn(
            torch_module, loss_kind, weight=ig_weight_tensor, focal_gamma=focal_gamma,
            log_priors=ig_log_priors, logit_adjust_tau=logit_adjust_tau,
            ignore_index=ignore_index, head_label="ig",
        )
        dc_to_ig = dc_to_ig.to(device)
        try:
            from vision_backend.training.hierarchy import map_dc_labels_to_ig
        except ModuleNotFoundError:
            from training.hierarchy import map_dc_labels_to_ig

    use_cuda_amp = bool(use_amp and device.type == "cuda")
    if amp_dtype not in ("fp16", "bf16"):
        raise ValueError(f"amp_dtype must be 'fp16' or 'bf16', got {amp_dtype!r}")
    autocast_dtype = torch_module.bfloat16 if amp_dtype == "bf16" else torch_module.float16
    # Loss scaling exists only to keep fp16 gradients out of the underflow range;
    # bf16 shares fp32's exponent range and must NOT be scaled.
    scaler = (
        torch_module.amp.GradScaler("cuda", enabled=True)
        if training and use_cuda_amp and amp_dtype == "fp16"
        else None
    )

    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=len(dataloader),
            desc=progress_desc or ("train" if training else "val"),
            unit="batch",
            dynamic_ncols=True,
            leave=leave_progress,
        )

    total_samples = 0
    total_loss = 0.0
    num_batches = len(dataloader)
    # Global confusion matrices, accumulated across the whole split. Averaging
    # per-batch mIoUs (the previous behaviour) is not the dataset mIoU: IoU is a
    # ratio of pixel counts, and each batch averaged over only the classes
    # present in it, so rare classes were weighted by their co-occurrence rate.
    # That number selected the best epoch, the best trial per variant, and the
    # v0-v3 ranking itself.
    conf = torch_module.zeros(
        (num_classes, num_classes), dtype=torch_module.long, device=device
    )
    conf_ig = (
        torch_module.zeros(
            (int(num_classes_ig), int(num_classes_ig)),
            dtype=torch_module.long, device=device,
        )
        if want_ig
        else None
    )

    grad_context = torch_module.enable_grad if training else torch_module.no_grad
    try:
        with grad_context():
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(dataloader):
                local, target, context = parse_segmentation_batch(batch)
                local = local.to(device, non_blocking=True).float()
                target = target.to(device, non_blocking=True).long()
                context_tensor = (
                    context.to(device, non_blocking=True).float()
                    if context is not None
                    else None
                )
                ig_target = (
                    map_dc_labels_to_ig(torch_module, target, dc_to_ig, ignore_index)
                    if want_ig
                    else None
                )

                batch_size = local.size(0)

                def _forward_and_loss():
                    if want_ig:
                        out = (
                            model(local, context_tensor, return_ig=True)
                            if context_tensor is not None
                            else model(local, return_ig=True)
                        )
                        dc_logits, ig_logits = out
                        if ig_logits is None:
                            raise RuntimeError(
                                "ig_loss_weight > 0 but the model has no IG head "
                                "(decoder.head_ig is None) -- build it with num_classes_ig set."
                            )
                        dc_loss = dc_loss_fn(dc_logits, target)
                        ig_loss = ig_loss_fn(ig_logits, ig_target)
                        return dc_logits, ig_logits, dc_loss + ig_loss_weight * ig_loss
                    dc_logits = model(local, context_tensor) if context_tensor is not None else model(local)
                    return dc_logits, None, dc_loss_fn(dc_logits, target)

                autocast_context = (
                    torch_module.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
                    if use_cuda_amp
                    else None
                )
                if autocast_context is None:
                    logits, ig_logits, loss = _forward_and_loss()
                else:
                    with autocast_context:
                        logits, ig_logits, loss = _forward_and_loss()

                if training:
                    # Average over the accumulation window so effective-batch
                    # gradients match a single step over accum_steps*batch_size
                    # samples -- keeps the LR/batch-size coupling the sweep found.
                    scaled_loss = loss / accum_steps
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    is_boundary = (
                        (batch_index + 1) % accum_steps == 0
                        or (batch_index + 1) == num_batches
                    )
                    if is_boundary:
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            # unscale_ first so the clip threshold is in real
                            # gradient units, not GradScaler-scaled ones.
                            # GradScaler.step below skips the update if this
                            # already found inf/nan, so clipping never hides a
                            # non-finite gradient -- it only bounds the finite
                            # ones, which is what actually stops the runaway
                            # weight growth that ends in NaN weights.
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            torch_module.nn.utils.clip_grad_norm_(
                                (p for g in optimizer.param_groups for p in g["params"]),
                                max_norm=grad_clip_norm,
                            )
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                        if ema is not None:
                            # F5: shadow weights updated once per optimizer
                            # step (not once per epoch) so they track the live
                            # model at the granularity the EMA formula assumes.
                            # ema_source_model lets the caller pass the plain
                            # (uncompiled) module when `model` is a
                            # torch.compile OptimizedModule, whose parameter
                            # names ModelEMA.update wouldn't otherwise match.
                            ema.update(ema_source_model if ema_source_model is not None else model)

                _accumulate_confusion(
                    torch_module, conf, logits, target,
                    num_classes=num_classes, ignore_index=ignore_index,
                )
                if want_ig:
                    _accumulate_confusion(
                        torch_module, conf_ig, ig_logits, ig_target,
                        num_classes=int(num_classes_ig), ignore_index=ignore_index,
                    )
                loss_value = loss.item()
                if error_on_nonfinite_loss and not np.isfinite(loss_value):
                    # Once the weights themselves go non-finite nothing recovers:
                    # every later epoch reports nan loss and a degenerate
                    # constant mIoU, and the best-checkpoint guard means nothing
                    # is ever saved again. Fail loudly here instead of burning
                    # the rest of the run producing garbage.
                    raise FloatingPointError(
                        f"non-finite {'train' if training else 'val'} loss "
                        f"({loss_value}) at batch {batch_index + 1}/{num_batches}. "
                        "Training diverged -- lower the learning rate and/or set "
                        "--grad-clip-norm. Pass error_on_nonfinite_loss=False to "
                        "continue anyway."
                    )
                total_samples += batch_size
                total_loss += loss_value * batch_size

                if progress is not None:
                    # Running loss only: mIoU is now a global quantity and is
                    # not defined mid-epoch without forcing a device sync.
                    progress.set_postfix(
                        loss=f"{total_loss / max(total_samples, 1):.4f}",
                    )
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    # One sync per epoch, here.
    global_metrics = _metrics_from_confusion(torch_module, conf)
    result = {
        "loss": total_loss / max(total_samples, 1),
        "pixel_acc": global_metrics["pixel_acc"],
        "miou": global_metrics["miou"],
    }
    if want_ig:
        # Extra keys only appear when F1 is active -- callers that don't use
        # it keep getting exactly today's 3-key dict.
        ig_metrics = _metrics_from_confusion(torch_module, conf_ig)
        result["pixel_acc_ig"] = ig_metrics["pixel_acc"]
        result["miou_ig"] = ig_metrics["miou"]
    return result


def tensor_to_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            normalized[key] = float(value.item())
        else:
            normalized[key] = float(value)
    return normalized


def flatten_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def infer_context_feature_channels(
    torch_module,
    encoder,
    *,
    local_input_size: int,
    context_input_size: int,
    in_channels: int = 1,
) -> list[int]:
    training = encoder.training
    try:
        param_device = next(encoder.parameters()).device
    except StopIteration:
        param_device = torch_module.device("cpu")
    encoder.eval()
    with torch_module.no_grad():
        local = torch_module.zeros(
            1,
            in_channels,
            local_input_size,
            local_input_size,
            device=param_device,
        )
        context = torch_module.zeros(
            1,
            in_channels,
            context_input_size,
            context_input_size,
            device=param_device,
        )
        features = encoder(local, context)
    encoder.train(training)
    return [int(feature.shape[1]) for feature in features]
