from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from jax_experiments.algos.sac_switch_recovery import SACSwitchRecovery
from jax_experiments.analysis import (
    regime_polarity_ant_switch_recovery_v24 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_ant_switch_recovery_v24 as trainer,
)
from scripts import submit_regime_polarity_ant_switch_recovery_v24 as submit


def _config():
    return SimpleNamespace(
        hidden_dim=8,
        ensemble_size=2,
        alpha=0.2,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
        seed=7,
        switch_recovery_target_mode=2,
        switch_recovery_segment_steps=5,
        switch_recovery_termination_penalty=10.0,
    )


class _FakeEnv:
    num_modes = 4

    def __init__(self):
        self.mode = 0
        self.calls = []

    def _activate_mode(self, mode):
        self.mode = int(mode)

    def rollout(self, params, n_steps, key, continue_state=False):
        del params, key
        self.calls.append((self.mode, int(n_steps), bool(continue_state)))
        value = float(self.mode)
        obs = jnp.full((n_steps, 3), value)
        act = jnp.full((n_steps, 2), value)
        rew = jnp.ones((n_steps,))
        next_obs = obs + 1.0
        done = jnp.zeros((n_steps,)).at[-1].set(
            1.0 if self.mode == 2 else 0.0)
        return (obs, act, rew, next_obs, done), []


class _FakeReplay:
    def push_batch_jax(self, *values):
        self.values = values


def test_predecessors_are_balanced_and_never_target():
    values = [
        trainer.predecessor_mode(2, 85003, 1400, cycle)
        for cycle in range(12)
    ]
    assert 2 not in values
    assert {mode: values.count(mode) for mode in (0, 1, 3)} == {
        0: 4, 1: 4, 3: 4,
    }


def test_collector_counts_physical_steps_and_applies_terminal_penalty():
    config = _config()
    agent = SACSwitchRecovery(3, 2, config, seed=config.seed)
    env = _FakeEnv()
    replay = _FakeReplay()
    rewards, rollout = trainer.collect_switch_recovery_samples(
        agent, env, replay, config, n_steps=20, current_iter=11)

    obs, _, rew, _, done, task_ids = replay.values
    assert obs.shape == (10, 3)
    assert rewards == [5.0, 5.0]
    assert rollout["physical_steps"] == 20
    assert rollout["candidate_steps"] == 10
    assert jnp.all(task_ids == 2)
    terminal_indices = jnp.asarray([4, 9])
    assert jnp.allclose(rew.reshape(-1)[terminal_indices], -9.0)
    assert jnp.allclose(done.reshape(-1)[terminal_indices], 1.0)
    expected_predecessors = [
        trainer.predecessor_mode(2, config.seed, 11, cycle)
        for cycle in range(2)
    ]
    assert [row[0] for row in env.calls] == [
        expected_predecessors[0], 2, expected_predecessors[1], 2,
    ]
    assert env.calls[0][2] is False
    assert all(row[2] is True for row in env.calls[1:])


def test_protocol_budget_and_variants_are_fixed():
    assert protocol.PHYSICAL_SAMPLES_PER_ITER == 8000
    assert protocol.CANDIDATE_SAMPLES_PER_ITER == 4000
    assert protocol.FINAL_TOTAL_STEPS == 11_200_000
    assert protocol.FINAL_UPDATE_COUNT == 525_000
    assert protocol.TERMINATION_PENALTY == {
        "switch_state": 0.0,
        "switch_state_risk": 500.0,
    }
    assert protocol.TRANSIENT_FALLBACK_STEPS == 8


def test_frozen_fallback_checkpoint_state_round_trips():
    config = _config()
    source = SACSwitchRecovery(3, 2, config, seed=11)
    restored = SACSwitchRecovery(3, 2, config, seed=29)
    restored.load_checkpoint_state(source.checkpoint_state())
    observations = jax.random.normal(jax.random.PRNGKey(31), (16, 3))
    assert jnp.allclose(
        source.fallback_policy.deterministic(observations),
        restored.fallback_policy.deterministic(observations),
    )


def test_scheduler_graph_has_linux_only_audits_and_no_local_gpu():
    rows = submit.candidates("high")
    assert len(rows) == 31
    specs = [spec for _, spec, _ in rows]
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(gpu) == 24
    assert len(cpu) == 7
    assert all("local" not in spec["allowed_nodes"] for spec in gpu)
    assert all(
        "jtl110gpu2" not in spec["allowed_nodes"] for spec in gpu)
    assert all(
        str(protocol.PRIOR_ANALYSIS_ROOT) in spec["stage_input_paths"]
        for spec in specs
    )
    assert all(
        set(spec["allowed_nodes"]) <= {
            "node001", "node002", "node003", "node004",
            "node005", "node006", "node007",
        }
        for spec in cpu
    )
