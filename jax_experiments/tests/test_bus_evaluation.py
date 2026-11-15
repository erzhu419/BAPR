import json
from types import SimpleNamespace

import numpy as np

from bus_evaluation import evaluate_bus_policy_paired, write_bus_evaluation


class FakePolicy:
    def __init__(self):
        self._warmup_active = True

    def get_action(self, state, deterministic):
        assert deterministic
        return np.asarray([0.0], dtype=np.float32)


class FakeBusEnv:
    max_agent_num = 1
    mode_switch_count = 1
    mode_history = [("normal", 0), ("demand_surge", 10)]

    def reset(self):
        self.steps = 0

    def initialize_state(self, render=False):
        return {0: [[0.0, 0.0]]}, {0: 0.0}, False

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            reward = float(np.random.normal())
            return {0: [[0.0, 0.0], [0.0, 1.0]]}, {0: reward}, False
        return {}, {}, True


def test_paired_bus_evaluation_replays_identical_event_seeds(tmp_path):
    trainer = SimpleNamespace(policy_net=FakePolicy())
    seeds = [100, 101, 102]
    first = evaluate_bus_policy_paired(trainer, FakeBusEnv, seeds)
    second = evaluate_bus_policy_paired(trainer, FakeBusEnv, seeds)

    assert trainer.policy_net._warmup_active is False
    assert first == second
    assert first["episode_count"] == 3
    assert [row["seed"] for row in first["episodes"]] == seeds
    assert all(row["mode_switch_count"] == 1 for row in first["episodes"])

    output = tmp_path / "paired_eval.json"
    write_bus_evaluation(output, first, {"algo": "fake"})
    payload = json.loads(output.read_text())
    assert payload["metadata"]["algo"] == "fake"
    assert payload["schema"] == "bapr.bus-paired-eval.v1"
