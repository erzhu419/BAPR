from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bus_experiments.frozen_policy_bank_core import (
    BusSAC,
    ReplayBuffer,
    TrainConfig,
    load_training_checkpoint,
    save_training_checkpoint,
)
from env.sim import env_bus


ROOT = Path(__file__).resolve().parents[2]


def _fake_environment():
    return SimpleNamespace(
        max_agent_num=8,
        stations=list(range(10)),
        timetables=[SimpleNamespace(launch_time=7200)],
        state_dim=7,
        action_space=SimpleNamespace(
            shape=(1,), high=np.asarray([60.0], dtype=np.float32)),
    )


def _small_config() -> TrainConfig:
    return TrainConfig(
        seed=7,
        max_episodes=3,
        role="robust_source",
        hidden_dim=8,
        ensemble_size=2,
        replay_capacity=32,
        batch_size=8,
        checkpoint_interval=1,
    )


def _fill_replay(buffer: ReplayBuffer) -> None:
    for index in range(12):
        state = np.asarray([
            index % 8, index % 5, index % 4, index % 2,
            300.0 + index, 340.0 - index, 20.0,
        ], dtype=np.float32)
        buffer.push(
            state, np.asarray([index % 60], dtype=np.float32),
            -float(index), state + np.asarray([0, 0, 0, 0, 1, -1, 0]),
            False)


def test_fixed_bus_mode_persists_and_conflicts_are_rejected():
    environment = env_bus(str(ROOT / "env"), fixed_mode="demand_surge")
    environment.reset()
    assert environment.current_mode_name == "demand_surge"
    assert environment.mode_history == [("demand_surge", 0)]
    assert environment.next_switch_time == float("inf")
    assert all(
        station.od_multiplier in {1.5, 3.0, 4.0, 5.0}
        for station in environment.stations)

    actions = {key: 0.0 for key in range(environment.max_agent_num)}
    for _ in range(500):
        environment.step(actions)
    assert environment.current_mode_name == "demand_surge"
    assert environment.mode_switch_count == 0

    with pytest.raises(ValueError, match="cannot both"):
        env_bus(
            str(ROOT / "env"), fixed_mode="normal",
            enable_mode_switch=True)


def test_bus_sac_update_and_checkpoint_roundtrip(tmp_path):
    config = _small_config()
    replay = ReplayBuffer(config.replay_capacity)
    _fill_replay(replay)
    trainer = BusSAC(_fake_environment(), config, torch.device("cpu"))
    diagnostics = trainer.update(replay, update_index=0)
    assert np.isfinite(diagnostics["critic_loss"])
    assert np.isfinite(diagnostics["policy_loss"])
    assert diagnostics["weighted_reg_mean"] > 0.0

    checkpoint_dir = tmp_path / "checkpoints"
    save_training_checkpoint(
        checkpoint_dir, trainer, replay, completed_episodes=2,
        total_steps=12, last_trained_step=12, update_index=1,
        rewards=[-10.0, -8.0], diagnostics=[diagnostics])

    restored_replay = ReplayBuffer(config.replay_capacity)
    restored = BusSAC(_fake_environment(), config, torch.device("cpu"))
    state = load_training_checkpoint(
        checkpoint_dir, restored, restored_replay)
    assert state["completed_episodes"] == 2
    assert state["update_index"] == 1
    assert len(restored_replay) == len(replay)
    for expected, actual in zip(
            trainer.policy.parameters(), restored.policy.parameters()):
        assert torch.equal(expected, actual)
