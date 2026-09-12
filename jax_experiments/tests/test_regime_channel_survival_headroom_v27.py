"""Tests for the V27 Hopper/Walker structured-channel headroom screen."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_channel_survival_headroom_v27 as analyzer,
)
from jax_experiments.analysis import (
    regime_channel_survival_headroom_v27 as protocol,
)
from jax_experiments.analysis import (
    run_regime_channel_survival_headroom_controller_v27 as controller,
)
from jax_experiments.envs.stochastic_mode_env import (
    MODE_FAMILIES,
    StochasticModeEnv,
)


def _submit_module():
    path = protocol.ROOT / "scripts/submit_regime_channel_survival_headroom_v27.py"
    spec = importlib.util.spec_from_file_location("_test_submit_v27", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_protocol_budget_and_disjoint_splits():
    protocol.assert_protocol_integrity()
    assert protocol.ENVS == ("Hopper-v2", "Walker2d-v2")
    assert protocol.FAMILY == "structured_channel"
    assert protocol.FINAL_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_UPDATE_COUNT == 350_000
    assert set(protocol.TRAINING_SEEDS).isdisjoint(protocol.AUDIT_EVENT_SEEDS)


def test_structured_channel_is_positive_invertible_for_odd_and_even_actions():
    profiles = MODE_FAMILIES[protocol.FAMILY]
    for act_dim in (3, 6):
        fake = SimpleNamespace(act_dim=act_dim)
        gains = [
            StochasticModeEnv._action_gain_for_profile(fake, profile)
            for profile in profiles
        ]
        assert all(np.all(gain > 0.0) for gain in gains)
        assert len({tuple(gain.tolist()) for gain in gains}) == 4


def test_controller_commands_differ_only_by_registered_context_role():
    robust = controller.training_command("Hopper-v2", "robust", 86_003)
    oracle = controller.training_command("Hopper-v2", "oracle", 86_003)
    assert robust[robust.index("--regime_context_source") + 1] == "robust"
    assert oracle[oracle.index("--regime_context_source") + 1] == "oracle"
    assert robust[robust.index("--stochastic_mode_family") + 1] == (
        protocol.FAMILY
    )
    assert robust[robust.index("--max_iters") + 1] == "1400"


def test_survival_gate_rejects_high_absolute_termination():
    row = {
        "switching_relative_gain": 0.20,
        "worst_mode_relative_gain": 0.20,
        "paired_deltas": {
            "switching": {"wins": 3},
            "stationary_worst": {"wins": 3},
        },
        "mode_wins": 4,
        "switching_termination_gap": 0.0,
        "summaries": {
            role: {
                "stationary_termination_mean": 0.0,
                "switching_termination_mean": 0.0,
            }
            for role in protocol.ROLES
        },
    }
    assert analyzer.apply_survival_gate(row.copy())["env_gate_pass"] is True
    row["summaries"]["oracle"]["stationary_termination_mean"] = 0.11
    assert analyzer.apply_survival_gate(row)["env_gate_pass"] is False


def test_scheduler_dag_is_remote_batched_and_dependency_gated():
    submit = _submit_module()
    rows = submit.candidates("high")
    train = [spec for _, spec, _ in rows if "/train/" in spec["signature"]]
    audits = [spec for _, spec, _ in rows if "/audit/" in spec["signature"]]
    aggregate = [
        spec for _, spec, _ in rows if spec["signature"].endswith("/analysis")
    ]
    assert len(rows) == 25
    assert len(train) == 12 and len(audits) == 12 and len(aggregate) == 1
    assert all("local" not in spec["allowed_nodes"] for spec in train)
    assert all(spec["vram"] == 2600 for spec in train)
    assert all(spec["wait_for_files"] for spec in audits + aggregate)
    assert all(spec["resume_managed_by_cmd"] is True for spec in train)
