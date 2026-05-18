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
    with torch_module.no_grad():
        preds = logits.argmax(dim=1)
        valid_mask = targets != ignore_index
        if valid_mask.sum() == 0:
            return {"pixel_acc": 0.0, "miou": 0.0}

        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        pixel_acc = correct / max(total, 1)

        ious: list[float] = []
        for class_index in range(num_classes):
            pred_mask = (preds == class_index) & valid_mask
            target_mask = (targets == class_index) & valid_mask
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            if union == 0:
                continue
            ious.append(intersection / union)

        miou = float(sum(ious) / max(len(ious), 1))
        return {"pixel_acc": float(pixel_acc), "miou": miou}


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
    progress_desc: Optional[str] = None,
    leave_progress: bool = False,
) -> dict[str, float]:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    training = optimizer is not None
    model.train(training)
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=ignore_index)
    use_cuda_amp = bool(use_amp and device.type == "cuda")
    scaler = (
        torch_module.amp.GradScaler("cuda", enabled=True)
        if training and use_cuda_amp
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
    total_pixel_acc = 0.0
    total_miou = 0.0

    grad_context = torch_module.enable_grad if training else torch_module.no_grad
    try:
        with grad_context():
            for batch in dataloader:
                local, target, context = parse_segmentation_batch(batch)
                local = local.to(device, non_blocking=True).float()
                target = target.to(device, non_blocking=True).long()
                context_tensor = (
                    context.to(device, non_blocking=True).float()
                    if context is not None
                    else None
                )

                batch_size = local.size(0)
                if training:
                    optimizer.zero_grad(set_to_none=True)

                autocast_context = (
                    torch_module.amp.autocast(device_type="cuda", enabled=True)
                    if use_cuda_amp
                    else None
                )
                if autocast_context is None:
                    logits = model(local, context_tensor) if context_tensor is not None else model(local)
                    loss = loss_fn(logits, target)
                else:
                    with autocast_context:
                        logits = model(local, context_tensor) if context_tensor is not None else model(local)
                        loss = loss_fn(logits, target)

                if training:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                metrics = _compute_segmentation_metrics(
                    torch_module,
                    logits,
                    target,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_pixel_acc += metrics["pixel_acc"] * batch_size
                total_miou += metrics["miou"] * batch_size

                if progress is not None:
                    progress.set_postfix(
                        loss=f"{total_loss / max(total_samples, 1):.4f}",
                        miou=f"{total_miou / max(total_samples, 1):.4f}",
                    )
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return {
        "loss": total_loss / max(total_samples, 1),
        "pixel_acc": total_pixel_acc / max(total_samples, 1),
        "miou": total_miou / max(total_samples, 1),
    }


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
