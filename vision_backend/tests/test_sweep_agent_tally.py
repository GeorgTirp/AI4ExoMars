"""maybe_run_sweep's end-of-agent verdict: a hyperband early stop is not a
failure, real errors still surface. Uses a fake wandb (no network)."""

from types import SimpleNamespace

import pytest

from vision_backend.training import wandb_utils
from vision_backend.training.wandb_utils import SweepConfigurationError, maybe_run_sweep


class _FakeRun:
    id, url = "r", "u"
    config = {"optimization.muon_lr": 1e-4}

    def finish(self):
        pass


class _FakeWandb:
    def init(self, **kwargs):
        return _FakeRun()

    def agent(self, sweep_id, function, count, **kwargs):
        for _ in range(count):
            try:
                function()
            except BaseException:  # wandb.agent swallows per-trial exceptions
                pass


def _run(monkeypatch, outcomes):
    monkeypatch.setattr(wandb_utils, "_import_wandb", lambda: _FakeWandb())
    monkeypatch.setattr(wandb_utils, "merge_wandb_config", lambda base, run: base)
    args = SimpleNamespace(
        wandb=True, no_wandb=False, wandb_mode="online", wandb_sweep_config=None,
        wandb_sweep_id="e/p/s", wandb_sweep_count=len(outcomes), wandb_project="p",
        wandb_entity=None, wandb_group=None, wandb_job_type=None, wandb_tags=None,
    )
    it = iter(outcomes)

    def train_fn(config, run):
        exc = next(it)
        if exc is not None:
            raise exc

    return maybe_run_sweep(args, stage_name="s", base_config={}, train_fn=train_fn)


def test_single_trial_early_stopped_by_hyperband_is_a_clean_exit(monkeypatch):
    assert _run(monkeypatch, [Exception()]) is True  # bare Exception = wandb's stop signal


def test_every_trial_of_a_multi_trial_agent_stopped_still_raises(monkeypatch):
    with pytest.raises(SweepConfigurationError, match="early-stopped"):
        _run(monkeypatch, [Exception(), Exception()])


def test_a_real_failure_still_raises(monkeypatch):
    with pytest.raises(SweepConfigurationError, match="failed"):
        _run(monkeypatch, [RuntimeError("boom")])


def test_completed_trial_is_a_clean_exit(monkeypatch):
    assert _run(monkeypatch, [None, Exception()]) is True
