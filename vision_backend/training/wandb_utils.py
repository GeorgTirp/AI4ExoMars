from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional


def _set_by_dotted_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if "." in key:
            _set_by_dotted_path(merged, key, value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def add_wandb_arguments(parser) -> None:
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="ai4exomars")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-job-type", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--wandb-sweep-config",
        default=None,
        help="Path to a JSON or YAML sweep configuration.",
    )
    parser.add_argument(
        "--wandb-sweep-id",
        default=None,
        help="Existing wandb sweep id to attach an agent to.",
    )
    parser.add_argument(
        "--wandb-sweep-count",
        type=int,
        default=None,
        help="Optional maximum number of runs when acting as a sweep agent.",
    )


def _import_wandb():
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install it with `uv add wandb` or "
            "`uv pip install wandb` before enabling wandb logging."
        ) from exc
    return wandb


def _load_sweep_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for non-JSON wandb sweep configs."
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("Sweep config must deserialize to a dictionary.")
    return loaded


def init_wandb_run(args, config: dict[str, Any], *, stage_name: str):
    if not args.wandb:
        return None

    wandb = _import_wandb()
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name,
        tags=args.wandb_tags,
        job_type=args.wandb_job_type or stage_name,
        mode=args.wandb_mode,
        config=config,
    )


def finish_wandb_run(run) -> None:
    if run is not None:
        run.finish()


def log_metrics(run, metrics: dict[str, Any], *, step: Optional[int] = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def merge_wandb_config(base_config: dict[str, Any], run) -> dict[str, Any]:
    if run is None:
        return base_config

    overrides = dict(run.config)
    return _deep_update(base_config, overrides)


def maybe_run_sweep(
    args,
    *,
    stage_name: str,
    base_config: dict[str, Any],
    train_fn: Callable[[dict[str, Any], Any], dict[str, Any]],
) -> bool:
    if not args.wandb_sweep_config and not args.wandb_sweep_id:
        return False

    wandb = _import_wandb()

    if not args.wandb:
        raise ValueError("Use `--wandb` together with sweep mode.")

    if args.wandb_sweep_id is not None:
        sweep_id = args.wandb_sweep_id
    else:
        sweep_path = Path(args.wandb_sweep_config).expanduser()
        sweep_config = _load_sweep_config(sweep_path)
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )

    def _agent_main():
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type=args.wandb_job_type or stage_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
        )
        try:
            merged_config = merge_wandb_config(base_config, run)
            train_fn(merged_config, run)
        finally:
            finish_wandb_run(run)

    wandb.agent(
        sweep_id,
        function=_agent_main,
        count=args.wandb_sweep_count,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )
    return True
