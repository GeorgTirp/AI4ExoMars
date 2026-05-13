# train_utils.py
"""
Training and evaluation utilities for semantic segmentation on AI4Mars.

This module provides:

- `_compute_batch_metrics`:
    Per-batch pixel accuracy and mean IoU computation.
- `train_one_epoch`:
    Single-epoch training loop with optional AMP, tqdm, and LR scheduler.
- `evaluate`:
    Validation/test loop mirroring the training metrics.
- `save_checkpoint` & `load_checkpoint`:
    Lightweight checkpoint helpers for model + optimizer + scheduler state.

All helpers assume a standard semantic segmentation setup with logits of shape
``[B, C, H, W]`` and integer masks of shape ``[B, H, W]``.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm

from .dataloader import AI4MARS_IGNORE_INDEX


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = AI4MARS_IGNORE_INDEX,
) -> Dict[str, float]:
    r"""
    Compute pixel accuracy and mean Intersection-over-Union (mIoU) for a batch.

    Given predictions :math:`\hat{y}` and ground-truth labels :math:`y`, we compute:

    - **Pixel accuracy**:

      .. math::

          \text{PixAcc} = \frac{\#\{\hat{y} = y,\, y \ne \text{ignore}\}}
                               {\#\{y \ne \text{ignore}\}}

    - **Class-wise IoU** for each class :math:`c \in \{0,\dots,K-1\}`:

      .. math::

          \mathrm{IoU}_c =
          \frac{|\{\hat{y}=c \land y=c\}|}
               {|\{\hat{y}=c \lor y=c\}|}

      and **mIoU** = average over classes that appear in the batch.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape ``[B, C, H, W]``.
    targets : torch.Tensor
        Ground-truth masks of shape ``[B, H, W]`` with integer class indices.
    num_classes : int
        Number of valid semantic classes.
    ignore_index : int, optional
        Label value to be masked out from metric computation.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:

        - ``"pixel_acc"`` : float
        - ``"miou"`` : float
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)  # [B,H,W]

        valid_mask = targets != ignore_index
        if valid_mask.sum() == 0:
            return {"pixel_acc": 0.0, "miou": 0.0}

        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        pixel_acc = correct / max(total, 1)

        ious = []
        for c in range(num_classes):
            pred_c = (preds == c) & valid_mask
            target_c = (targets == c) & valid_mask

            intersection = (pred_c & target_c).sum().item()
            union = (pred_c | target_c).sum().item()

            if union == 0:
                continue
            ious.append(intersection / union)

        miou = float(sum(ious) / max(len(ious), 1))
        return {"pixel_acc": float(pixel_acc), "miou": miou}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = False,
    use_tqdm: bool = False,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
    scheduler: Optional[_LRScheduler] = None,
) -> Dict[str, float]:
    r"""
    Train a segmentation model for a single epoch.

    The loop supports:

    - Mixed precision via ``torch.cuda.amp`` (``use_amp=True``).
    - A per-step LR scheduler (e.g. cosine with warmup).
    - Optional tqdm progress bars.

    Parameters
    ----------
    model : nn.Module
        Segmentation model producing logits of shape ``[B, C, H, W]``.
    dataloader : torch.utils.data.DataLoader
        Training data loader yielding ``(images, masks)`` batches.
    optimizer : torch.optim.Optimizer
        Optimizer instance (e.g. Muon, NAdam, AdamW).
    device : torch.device
        Target device (e.g. ``torch.device("cuda")``).
    num_classes : int
        Number of semantic classes (for mIoU computation).
    loss_fn : nn.Module, optional
        Loss function. Defaults to ``nn.CrossEntropyLoss`` with
        ``ignore_index=AI4MARS_IGNORE_INDEX``.
    use_amp : bool, optional
        If ``True``, run the forward/backward pass in mixed precision.
    use_tqdm : bool, optional
        If ``True``, wrap the dataloader with a tqdm progress bar.
    epoch : int, optional
        Current epoch index (1-based), used only for labeling tqdm.
    num_epochs : int, optional
        Total number of epochs, used only for labeling tqdm.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Optional per-step LR scheduler; ``scheduler.step()`` is called once
        per batch.

    Returns
    -------
    Dict[str, float]
        Dictionary with average metrics over the epoch:

        - ``"loss"`` : mean training loss
        - ``"pixel_acc"`` : mean pixel accuracy
        - ``"miou"`` : mean IoU
    """
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=AI4MARS_IGNORE_INDEX)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    running_loss = 0.0
    total_samples = 0
    acc_sum = 0.0
    miou_sum = 0.0

    # Wrap dataloader with tqdm if requested
    if use_tqdm:
        if epoch is not None and num_epochs is not None:
            desc = f"Train Epoch {epoch}/{num_epochs}"
        else:
            desc = "Training"
        data_iter = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    else:
        data_iter = dataloader

    for imgs, masks in data_iter:
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()

        batch_size = imgs.size(0)
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size

        metrics = _compute_batch_metrics(logits, masks, num_classes)
        acc_sum += metrics["pixel_acc"] * batch_size
        miou_sum += metrics["miou"] * batch_size

        if use_tqdm:
            avg_loss_so_far = running_loss / max(total_samples, 1)
            avg_miou_so_far = miou_sum / max(total_samples, 1)
            data_iter.set_postfix(
                loss=f"{avg_loss_so_far:.3f}",
                mIoU=f"{avg_miou_so_far:.3f}",
            )

    avg_loss = running_loss / max(total_samples, 1)
    avg_acc = acc_sum / max(total_samples, 1)
    avg_miou = miou_sum / max(total_samples, 1)

    return {"loss": avg_loss, "pixel_acc": avg_acc, "miou": avg_miou}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_tqdm: bool = False,
) -> Dict[str, float]:
    r"""
    Evaluate a segmentation model on a validation or test split.

    This mirrors :func:`train_one_epoch` but:

    - Disables gradient computation.
    - Does **not** update model, optimizer, or scheduler.
    - Still reports mean loss, pixel accuracy, and mIoU.

    Parameters
    ----------
    model : nn.Module
        Segmentation model to evaluate.
    dataloader : torch.utils.data.DataLoader
        Validation or test data loader.
    device : torch.device
        Evaluation device.
    num_classes : int
        Number of semantic classes.
    use_tqdm : bool, optional
        If ``True``, show a tqdm progress bar over batches.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:

        - ``"loss"`` : mean loss,
        - ``"pixel_acc"`` : mean pixel accuracy,
        - ``"miou"`` : mean IoU.
    """
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=AI4MARS_IGNORE_INDEX)

    running_loss = 0.0
    total_samples = 0
    acc_sum = 0.0
    miou_sum = 0.0

    if use_tqdm:
        data_iter = tqdm(dataloader, desc="Eval", unit="batch", leave=False)
    else:
        data_iter = dataloader

    with torch.no_grad():
        for imgs, masks in data_iter:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).long()

            batch_size = imgs.size(0)
            total_samples += batch_size

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * batch_size

            metrics = _compute_batch_metrics(logits, masks, num_classes)
            acc_sum += metrics["pixel_acc"] * batch_size
            miou_sum += metrics["miou"] * batch_size

            if use_tqdm:
                avg_loss_so_far = running_loss / max(total_samples, 1)
                avg_miou_so_far = miou_sum / max(total_samples, 1)
                data_iter.set_postfix(
                    loss=f"{avg_loss_so_far:.3f}",
                    mIoU=f"{avg_miou_so_far:.3f}",
                )

    avg_loss = running_loss / max(total_samples, 1)
    avg_acc = acc_sum / max(total_samples, 1)
    avg_miou = miou_sum / max(total_samples, 1)

    return {"loss": avg_loss, "pixel_acc": avg_acc, "miou": avg_miou}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    r"""
    Save a training checkpoint to disk.

    The checkpoint is a ``dict`` that can contain:

    - ``"model_state"`` (always present)
    - ``"optimizer_state"`` (if ``optimizer`` is provided)
    - ``"scheduler_state"`` (if ``scheduler`` is provided)
    - ``"epoch"`` (if provided)
    - ``"metrics"`` (if provided)
    - ``"extra"`` (any additional user metadata)

    Parameters
    ----------
    path : str
        File path to save to (e.g. ``"checkpoints/best_model.pt"``).
    model : nn.Module
        Model whose ``state_dict`` will be stored as ``"model_state"``.
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state will be stored as ``"optimizer_state"``.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler whose state will be stored as ``"scheduler_state"``.
    epoch : int, optional
        Epoch index at the time of saving.
    metrics : Dict[str, float], optional
        Dictionary of scalar metrics (e.g. best validation mIoU).
    extra : Dict[str, Any], optional
        Arbitrary additional metadata to store.

    Returns
    -------
    None
    """
    state: Dict[str, Any] = {
        "model_state": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if metrics is not None:
        state["metrics"] = metrics
    if extra is not None:
        state["extra"] = extra

    torch.save(state, path)
    print(f"[checkpoint] Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    r"""
    Load a training checkpoint from disk and restore model/optimizer/scheduler.

    Parameters
    ----------
    path : str
        Path to the checkpoint file (e.g. ``"checkpoints/best_model.pt"``).
    model : nn.Module
        Model into which ``"model_state"`` will be loaded.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to restore from ``"optimizer_state"`` (if present).
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler to restore from ``"scheduler_state"`` (if present).
    map_location : str or torch.device, optional
        Device mapping for loaded tensors, passed directly to ``torch.load``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with any extra information stored in the checkpoint, with keys:

        - ``"epoch"`` : int or ``None``
        - ``"metrics"`` : dict or ``None``
        - ``"extra"`` : dict or ``None``

    Notes
    -----
    - Missing optimizer or scheduler states are silently ignored.
    - This function does **not** perform any strict version checking.
    """
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])
    print(f"[checkpoint] Loaded model weights from {path}")

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("[checkpoint] Restored optimizer state.")

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("[checkpoint] Restored scheduler state.")

    info: Dict[str, Any] = {
        "epoch": checkpoint.get("epoch", None),
        "metrics": checkpoint.get("metrics", None),
        "extra": checkpoint.get("extra", None),
    }
    return info
