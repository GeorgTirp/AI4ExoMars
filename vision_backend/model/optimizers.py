# optimizers.py
"""
Optimizer helpers for model training.

This module provides:

1. **create_optimizer**  
   Smart optimizer selection with priority:
   - Muon (if installed and explicitly enabled),
   - NAdam (PyTorch's NAdamW-like implementation),
   - AdamW as a safe fallback.

2. **create_cosine_scheduler_with_warmup**  
   A cosine–annealing learning rate schedule with linear warmup,
   mathematically equivalent to the Hugging Face transformers scheduler.

Both utilities are framework-agnostic and work with any PyTorch model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# Parameter-name fragments that must never receive weight decay even though
# they may be multi-dimensional (relative-position bias tables, GRN affine,
# the SimMIM mask fill scalar). 1-D params (norms/biases) are excluded by shape.
_NO_DECAY_NAME_FRAGMENTS = (
    "relative_position_bias_table",
    "mask_value",
    ".grn.",
)

# Layer-wise LR decay (LLRD) depth ranks, shallow (0) -> deep (_LLRD_MAX_DEPTH).
# Matches HybridEncoder's own component naming (model/hybrid_encoder.py:
# stem, s1, down1, s2, down2, s3, down3, s4, norm) exactly, so this is a name
# match, not a guess -- but it does NOT cover the legacy two-branch
# ContextAwareConvNeXtSwinEncoder (different internal naming); any "encoder."
# param that doesn't match one of these prefixes is conservatively treated as
# the shallowest rank (0, maximally discounted) since its true depth isn't
# known here -- safer than accidentally leaving an unrecognised deep-encoder
# component at full LR. Decoder + heads (anything not under "encoder.") always
# get the top rank, i.e. the full, undiscounted base LR.
_LLRD_ENCODER_PREFIXES: tuple[tuple[str, int], ...] = (
    ("encoder.stem", 0),
    ("encoder.s1", 1),
    ("encoder.down1", 2),
    ("encoder.s2", 3),
    ("encoder.down2", 4),
    ("encoder.s3", 5),
    ("encoder.down3", 6),
    ("encoder.s4", 7),
    ("encoder.norm", 7),
)
_LLRD_MAX_DEPTH = 8  # decoder/heads rank


def llrd_depth_rank(param_name: str) -> int:
    """Depth rank for `param_name`: 0 (stem, most LR-discounted) ..
    _LLRD_MAX_DEPTH (decoder/heads, full LR)."""
    for prefix, rank in _LLRD_ENCODER_PREFIXES:
        if param_name == prefix or param_name.startswith(prefix + "."):
            return rank
    if param_name.startswith("encoder."):
        return 0
    return _LLRD_MAX_DEPTH


def llrd_lr_scale(param_name: str, llrd: float) -> float:
    """Multiplier on a param's base LR for layer-wise LR decay. llrd=1.0 (the
    neutral default) always returns 1.0 regardless of depth -- current
    behavior, unchanged."""
    if llrd == 1.0:
        return 1.0
    return llrd ** (_LLRD_MAX_DEPTH - llrd_depth_rank(param_name))


def _group_params_by_llrd_rank(
    named_params: list[tuple[str, torch.nn.Parameter]],
    base_lr: float,
    llrd: float,
    *,
    extra: Optional[dict] = None,
) -> list[dict]:
    """Bucket (name, param) pairs by LLRD depth rank and build optimizer
    param-group dicts, each with its own `lr = base_lr * llrd_lr_scale(...)`.
    `extra` (e.g. {"weight_decay": ...}) is merged into every group unchanged
    -- LLRD only ever adds an `lr` axis on top of the existing decay grouping,
    never changes which bucket/optimizer a param belongs to."""
    by_rank: dict[int, list[torch.nn.Parameter]] = {}
    for name, param in named_params:
        by_rank.setdefault(llrd_depth_rank(name), []).append(param)
    groups = []
    for rank in sorted(by_rank):
        scale = 1.0 if llrd == 1.0 else llrd ** (_LLRD_MAX_DEPTH - rank)
        group = {"params": by_rank[rank], "lr": base_lr * scale}
        if extra:
            group.update(extra)
        groups.append(group)
    return groups


# Try importing Muon (optional dependency). This repo trains single-process,
# single-GPU (no torch.distributed.init_process_group anywhere), so we want
# SingleDeviceMuon -- the upstream `Muon` class now shards across ranks via
# dist.get_world_size()/all_gather and raises without a process group.
try:
    from muon import SingleDeviceMuon as Muon  # KellerJordan/Muon optimizer
    _HAS_MUON = True
except Exception:
    Muon = None  # type: ignore
    _HAS_MUON = False


# ---------------------------------------------------------------------------
# Optimizer Factory
# ---------------------------------------------------------------------------
def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    use_muon: bool = True,
    llrd: float = 1.0,
) -> torch.optim.Optimizer:
    r"""
    Create an optimizer for a given model with prioritized fallback logic.

    The optimizers are tried in the following priority:

    1. **Muon** (if installed and ``use_muon=True``)
       Muon is a second-order optimizer approximating natural gradient steps.

    2. **NAdam**
       PyTorch's NAdam implementation (NadamW-style), supporting weight decay.

    3. **AdamW**
       Stable, widely used, standard fallback.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose trainable parameters will be optimized.
    lr : float, optional
        Learning rate (default: ``3e-4``).
    weight_decay : float, optional
        Weight decay coefficient (default: ``1e-2``).
    use_muon : bool, optional
        Whether the user prefers to use Muon if available.
    llrd : float, optional
        Layer-wise LR decay factor (see `llrd_depth_rank`). ``1.0`` (default)
        gives every parameter the same `lr`, reproducing today's behavior
        exactly; ``<1`` discounts shallower encoder components more.

    Returns
    -------
    torch.optim.Optimizer
        Constructed optimizer instance.

    Notes
    -----
    - Only parameters with ``requires_grad=True`` are passed to the optimizer.
    - If Muon is requested but not installed, AdamW is used and a warning printed.
    """
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = _group_params_by_llrd_rank(named_params, lr, llrd)

    # --------------------
    # 1) Try Muon
    # --------------------
    if use_muon and _HAS_MUON:
        print("[optimizers] Using Muon optimizer.")
        return Muon(param_groups, lr=lr, weight_decay=weight_decay)  # type: ignore

    # --------------------
    # 2) Try NAdam (NadamW-style)
    # --------------------
    if hasattr(torch.optim, "NAdam"):
        print("[optimizers] Using NAdam (NadamW-style) optimizer.")
        return torch.optim.NAdam(param_groups, lr=lr, weight_decay=weight_decay)

    # --------------------
    # 3) Fallback: AdamW
    # --------------------
    if use_muon and not _HAS_MUON:
        print(
            "[optimizers] Muon requested but not installed.\n"
            "Install via: pip install git+https://github.com/KellerJordan/Muon"
        )

    print("[optimizers] Using AdamW optimizer.")
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Cosine Annealing LR Schedule with Warmup
# ---------------------------------------------------------------------------
def create_cosine_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""
    Create a cosine-annealing LR scheduler with linear warmup.

    This scheduler combines a linear warmup phase with a cosine decay phase.

    **Learning rate schedule**

    Given current step :math:`t`, warmup :math:`W`, and total steps :math:`T`,
    the schedule is:

    **Warmup (linear)**

    .. math::

        \text{lr}(t) = \frac{t}{W}, \quad 0 \le t < W

    **Cosine decay**

    .. math::

        \text{progress} = \frac{t - W}{T - W}

        \text{lr}(t) =
        \tfrac{1}{2}\left(1 + \cos\big( 2\pi \cdot C \cdot \text{progress} \big)\right)

    where:

    - :math:`C` = ``num_cycles`` controls the number of cosine waves
      (``0.5`` = standard: decay → 0 once)

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    num_warmup_steps : int
        Number of linear warmup steps, typically 5–10% of total training steps.
    num_training_steps : int
        Total number of steps (``epochs * steps_per_epoch``).
    num_cycles : float, optional
        Number of cosine cycles.
        Default ``0.5`` = half-cycle (decay to 0 exactly once).

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler that updates the LR dynamically during training.

    Notes
    -----
    - This implementation is mathematically similar to
      ``transformers.get_cosine_schedule_with_warmup``.
    - The value returned by the lambda is multiplied with the optimizer's base LR.
    """

    def lr_lambda(current_step: int) -> float:
        # ---- Linear warmup ----
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)

        # ---- Cosine decay ----
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Stage-1 SimMIM: weight-decay param groups + optimizer + floored scheduler
# ---------------------------------------------------------------------------
def split_decay_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Two param groups: weight decay on >=2-D weights, none on the rest.

    Excludes norms/biases (1-D), relative-position-bias tables, GRN affine
    params, and the mask fill scalar from weight decay (§5).
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(frag in name for frag in _NO_DECAY_NAME_FRAGMENTS):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_simmim_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    optimizer: str = "adamw",
) -> torch.optim.Optimizer:
    """AdamW (default) with WD param groups, or a Muon-hybrid behind a flag.

    The Muon-hybrid path is best-effort: it requires the optional ``muon``
    package and falls back to AdamW (with a warning) when it is unavailable.
    """
    param_groups = split_decay_param_groups(model, weight_decay)

    if optimizer == "muon_hybrid":
        if _HAS_MUON:
            try:
                print("[optimizers] Using Muon-hybrid (Muon on 2-D weights, AdamW elsewhere).")
                param_groups[0]["use_muon"] = True
                param_groups[1]["use_muon"] = False
                return Muon(param_groups, lr=lr, betas=betas)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - depends on muon version
                print(f"[optimizers] Muon-hybrid unavailable ({exc}); falling back to AdamW.")
        else:
            print("[optimizers] Muon requested but not installed; using AdamW.")

    print("[optimizers] Using AdamW with weight-decay param groups.")
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


# ---------------------------------------------------------------------------
# Routed hybrid: Muon on 2-D weight matrices, NAdam on everything else
# ---------------------------------------------------------------------------
class CombinedOptimizer(torch.optim.Optimizer):
    """Drive several sub-optimizers as one, so the existing single-optimizer
    training loop (AMP GradScaler, one LambdaLR, grad-clip) needs no changes.

    ``param_groups`` concatenates the sub-optimizers' groups (the SAME dict
    objects, not copies), so one LambdaLR scales every group by the same factor
    while preserving each group's independent base LR, and ``GradScaler`` /
    ``clip_grad_norm_`` see every parameter. ``step`` / ``zero_grad`` fan out.
    Subclasses ``Optimizer`` only to satisfy ``LRScheduler``'s isinstance check;
    the base ``__init__`` is intentionally not called.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer.")
        self.optimizers = list(optimizers)
        self.defaults = dict(self.optimizers[0].defaults)
        # concrete list of the real group dicts so scheduler lr writes propagate
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]

    @property
    def state(self):  # merged read-only view (unused by GradScaler/LambdaLR)
        merged: dict = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    # Muon flattens >=3-D updates with `update.view(len(update), -1)`.
                    # Under channels_last the gradient of a conv weight is not
                    # contiguous in NCHW order, so that view() raises
                    # "view size is not compatible with input tensor's size and
                    # stride". Beyond the crash it is a correctness issue: had the
                    # view succeeded it would have flattened in NHWC order, handing
                    # Newton-Schulz a differently-permuted matrix than intended.
                    # Normalising to contiguous makes the flattening layout-
                    # independent; it is a no-op for already-contiguous grads.
                    if p.grad is not None and p.grad.dim() >= 3 and not p.grad.is_contiguous():
                        p.grad = p.grad.contiguous()
            opt.step()
        return loss

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        for opt, sub in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sub)


def build_routed_muon_nadam_optimizer(
    model: torch.nn.Module,
    *,
    muon_lr: float,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    nadam_lr: float,
    nadam_betas: tuple[float, float] = (0.9, 0.999),
    nadam_weight_decay: float = 0.0,
    require_muon: bool = True,
    muon_scope: str = "matrix",
    llrd: float = 1.0,
) -> CombinedOptimizer:
    """Muon on weight matrices, NAdam on everything else.

    ``muon_scope`` controls what "matrix" means, because weight-decay grouping
    and optimizer routing are *different* questions that happen to share a
    predicate:

    - ``"matrix"`` (default): only genuinely 2-D weights -- attention qkv/proj
      and MLP layers -- go to Muon. Convolution weights are 4-D and go to NAdam
      (still with weight decay). This is the regime Muon is designed and
      benchmarked for.
    - ``"all"``: every >=2-D weight goes to Muon, reproducing the original
      routing. On this hybrid conv/transformer encoder that sent 48 of 80
      tensors -- every conv, including depthwise ones -- through Muon's
      Newton-Schulz orthogonalisation. Muon flattens a 4-D filter bank via
      ``update.view(len(update), -1)``, so a depthwise ``[C,1,7,7]`` becomes
      ``[C,49]`` and orthogonalisation forces mutually-orthogonal filters across
      channels that are independent by construction -- a constraint with no
      meaning for depthwise convolution, and outside Muon's validated domain.

    Muon and NAdam are independent optimizers with their own LR and momentum,
    wrapped in a CombinedOptimizer so the training loop is unchanged.

    ``llrd`` (see `llrd_depth_rank`) scales *within* each existing bucket --
    it never moves a param between the Muon/NAdam-decayed/NAdam-undecayed
    groups above, only multiplies that bucket's own base LR by a depth-
    dependent factor. ``1.0`` (default) reproduces today's LRs exactly.
    """
    if muon_scope not in ("matrix", "all"):
        raise ValueError(f"muon_scope must be 'matrix' or 'all', got {muon_scope!r}")

    muon_named: list[tuple[str, torch.nn.Parameter]] = []
    aux_decay_named: list[tuple[str, torch.nn.Parameter]] = []
    aux_no_decay_named: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(frag in name for frag in _NO_DECAY_NAME_FRAGMENTS):
            aux_no_decay_named.append((name, param))
        elif muon_scope == "all" or param.ndim == 2:
            muon_named.append((name, param))
        else:
            # >=3-D (conv) weights: not Muon's domain, but still decayed.
            aux_decay_named.append((name, param))

    aux_groups = (
        _group_params_by_llrd_rank(aux_no_decay_named, nadam_lr, llrd, extra={"weight_decay": nadam_weight_decay})
        + _group_params_by_llrd_rank(aux_decay_named, nadam_lr, llrd, extra={"weight_decay": muon_weight_decay})
    )
    print(f"[optimizers] muon_scope={muon_scope}: Muon {len(muon_named)} tensors, "
          f"NAdam {len(aux_decay_named)} conv/>=3-D (decayed) + {len(aux_no_decay_named)} 1-D/excluded. "
          f"llrd={llrd}")
    aux = torch.optim.NAdam(
        aux_groups, lr=nadam_lr, betas=nadam_betas, weight_decay=nadam_weight_decay
    )

    muon_groups = _group_params_by_llrd_rank(muon_named, muon_lr, llrd)

    if _HAS_MUON:
        print("[optimizers] Routed: Muon on 2-D weight matrices "
              f"(lr={muon_lr}, momentum={muon_momentum}), NAdam on the rest "
              f"(lr={nadam_lr}, betas={nadam_betas}).")
        muon = Muon(
            muon_groups, lr=muon_lr, momentum=muon_momentum,
            weight_decay=muon_weight_decay,
        )
        return CombinedOptimizer([muon, aux])

    if require_muon:
        raise RuntimeError(
            "Routed Muon+NAdam requested but the 'muon' package is not installed.\n"
            "Install it on the server:  uv pip install git+https://github.com/KellerJordan/Muon\n"
            "(or run with --use-muon off to use a single NAdam optimizer)."
        )

    # test/CI fallback only: NAdam on the weight group too (keeps plumbing usable
    # without Muon; NOT for real training -- Muon LRs would be far too large).
    # Rebuilt from nadam_lr (not muon_groups, which is scaled from muon_lr) --
    # that mismatch would silently feed muon-scale LRs into this NAdam.
    print("[optimizers] Muon unavailable -- FALLBACK: NAdam on the weight group "
          "too (plumbing only, do not train seriously like this).")
    muon_groups_fallback = _group_params_by_llrd_rank(muon_named, nadam_lr, llrd)
    muon_fallback = torch.optim.NAdam(
        muon_groups_fallback, lr=nadam_lr, betas=nadam_betas, weight_decay=muon_weight_decay
    )
    return CombinedOptimizer([muon_fallback, aux])


def create_warmup_cosine_with_floor(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay from 1.0 down to ``min_lr_ratio``.

    ``min_lr_ratio = min_lr / base_lr`` so the LR floors at ``min_lr`` instead
    of decaying to zero.
    """
    min_lr_ratio = max(0.0, min(min_lr_ratio, 1.0))

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
