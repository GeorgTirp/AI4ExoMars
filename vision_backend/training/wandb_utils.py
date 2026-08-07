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


class SweepConfigurationError(RuntimeError):
    """Raised when sweep flags are inconsistent — never swallowed."""


def sweep_requested(args) -> bool:
    return bool(getattr(args, "wandb_sweep_config", None) or getattr(args, "wandb_sweep_id", None))


def validate_wandb_args(args) -> None:
    """Reject flag combinations that would silently degrade to a non-sweep run.

    The failure mode this guards against: ``--wandb-sweep-id`` is accepted by the
    parser, so a launcher that forgets ``--wandb`` (or passes ``--no-wandb``, or
    an empty ``$SWEEP_ID``) would train exactly one run at the YAML defaults and
    report success. Every case below is fatal instead.
    """
    if not sweep_requested(args):
        if getattr(args, "wandb_sweep_count", None) is not None:
            raise SweepConfigurationError(
                "--wandb-sweep-count was given without --wandb-sweep-id/"
                "--wandb-sweep-config; this would run a single ordinary run."
            )
        return

    if getattr(args, "no_wandb", False):
        raise SweepConfigurationError("--no-wandb cannot be combined with sweep mode.")
    if not getattr(args, "wandb", False):
        raise SweepConfigurationError(
            "Sweep mode requires --wandb. Without it the sweep flags are inert "
            "and you would get one run at the config defaults."
        )
    if getattr(args, "wandb_mode", "online") != "online":
        raise SweepConfigurationError(
            f"Sweep mode requires --wandb-mode online (got "
            f"{getattr(args, 'wandb_mode', None)!r}); an agent cannot receive "
            f"hyperparameters from the sweep server otherwise."
        )
    if args.wandb_sweep_config and args.wandb_sweep_id:
        raise SweepConfigurationError(
            "Pass either --wandb-sweep-config (create a sweep) or "
            "--wandb-sweep-id (join one), not both."
        )
    if args.wandb_sweep_config:
        path = Path(args.wandb_sweep_config).expanduser()
        if not path.is_file():
            raise SweepConfigurationError(f"Sweep config not found: {path}")
    if args.wandb_sweep_id is not None and not args.wandb_sweep_id.strip():
        raise SweepConfigurationError(
            "--wandb-sweep-id is empty (an unset $SWEEP_ID in the launcher?)."
        )


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
    validate_wandb_args(args)
    if not sweep_requested(args):
        return False

    wandb = _import_wandb()

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

    print(f"[sweep] stage={stage_name} sweep_id={sweep_id} "
          f"project={args.wandb_project} entity={args.wandb_entity or '<default>'} "
          f"count={args.wandb_sweep_count if args.wandb_sweep_count is not None else 'unbounded'}",
          flush=True)

    tally = {"started": 0, "completed": 0}
    first_error: list[BaseException] = []

    def _agent_main():
        tally["started"] += 1
        trial = tally["started"]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type=args.wandb_job_type or stage_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
        )
        try:
            sampled = dict(run.config)
            print(f"[sweep] trial {trial} run={run.id} url={run.url}", flush=True)
            print(f"[sweep] trial {trial} sampled params: {sampled}", flush=True)
            if not sampled:
                raise SweepConfigurationError(
                    f"Sweep trial {trial} received an EMPTY parameter set from the "
                    f"sweep server. The agent is not actually tuning anything — "
                    f"check that sweep {sweep_id} exists and its `parameters` block "
                    f"is non-empty."
                )
            merged_config = merge_wandb_config(base_config, run)
            train_fn(merged_config, run)
            tally["completed"] += 1
            print(f"[sweep] trial {trial} completed ({tally['completed']}/{trial} ok)", flush=True)
        except BaseException as exc:  # record, then let wandb mark the run crashed
            if not first_error:
                first_error.append(exc)
            print(f"[sweep] trial {trial} FAILED: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            finish_wandb_run(run)

    wandb.agent(
        sweep_id,
        function=_agent_main,
        count=args.wandb_sweep_count,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )

    # wandb.agent swallows per-trial exceptions and returns normally, so an agent
    # whose every trial died would otherwise exit 0 and look like a clean run.
    if tally["started"] == 0:
        raise SweepConfigurationError(
            f"Sweep agent for {sweep_id} exited without starting a single trial. "
            f"The sweep is likely already finished/cancelled, or the id belongs to "
            f"another project/entity."
        )
    if tally["completed"] == 0:
        raise SweepConfigurationError(
            f"All {tally['started']} sweep trial(s) failed for {sweep_id}. "
            f"First error was {type(first_error[0]).__name__}: {first_error[0]}"
        ) from (first_error[0] if first_error else None)

    print(f"[sweep] agent done: {tally['completed']}/{tally['started']} trial(s) completed.",
          flush=True)
    return True
