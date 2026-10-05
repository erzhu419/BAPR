"""Source reward formula, actual update/CLI, contrasts and scheduler dependencies."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import pytest

from jax_experiments import train
from jax_experiments.algos.resac_torch import TorchEquationRESAC, bus_reward
from jax_experiments.analysis import resac_jax_reward_transfer_protocol_v45 as p
from jax_experiments.analysis import run_resac_jax_reward_transfer_v45 as run
from jax_experiments.configs.default import Config
from scripts import submit_resac_jax_reward_transfer_v45 as submit


@pytest.mark.parametrize("rewards", ([1., 3., 7., 11.], [4., 4., 4., 4.], [1e-8, 2e-8, 3e-8, 4e-8]))
def test_bus_formula_uses_sample_std_and_original_fallback(rewards):
    reward = np.asarray(rewards, dtype=np.float32)[:, None]
    actual = bus_reward(jnp.asarray(reward))
    std = np.std(reward, axis=0, ddof=1)
    expected = 10 * (reward - reward.mean(axis=0)) / np.where(std < 1e-6, 1, std)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("transform", ("raw", "bus_standardized"))
def test_update_uses_transformed_target_and_per_minibatch_scan_axis(transform):
    config = Config(algo="resac_torch", hidden_dim=8, ensemble_size=3, gamma=0,
                    beta_ood=0, resac_critic_actor_ratio=2, resac_reward_transform=transform)
    agent = TorchEquationRESAC(3, 2, config, seed=41)
    rewards = np.asarray([[-5., 0., 1., 3., 5., 10.], [90., 94., 95., 97., 99., 104.]], dtype=np.float32)[..., None]
    rng = np.random.default_rng(17)
    batch = {name: jnp.asarray(rng.normal(size=(2, 6, size)), dtype=jnp.float32)
             for name, size in (("obs", 3), ("act", 2), ("next_obs", 3))}
    batch.update(rew=jnp.asarray(rewards), done=jnp.zeros((2, 6, 1)))
    initial_q = np.asarray(agent.critic(batch["obs"][0], batch["act"][0]))
    target = np.asarray(bus_reward(batch["rew"][0]) if transform == "bus_standardized" else batch["rew"][0])
    first = {name: value[:1] for name, value in batch.items()}
    metrics = agent.multi_update(first)
    assert metrics["critic_loss"] == pytest.approx(float(((initial_q - target[:, 0]) ** 2).mean()), rel=1e-5)
    agent = TorchEquationRESAC(3, 2, config, seed=41)
    metrics = agent.multi_update(batch)
    np.testing.assert_array_equal(batch["rew"], rewards)
    if transform == "bus_standardized":
        assert metrics["training_reward_mean"] == pytest.approx(0, abs=1e-5)
        assert metrics["training_reward_std"] == pytest.approx(10, abs=1e-5)
    else:
        assert metrics["training_reward_mean"] == metrics["raw_reward_mean"]
        assert metrics["training_reward_std"] == metrics["raw_reward_std"]
    assert metrics["actor_update_rate"] == 0.5


@pytest.mark.parametrize("arm", p.ARMS)
def test_actual_cli_and_equal_budgets(arm, monkeypatch):
    command = run.training_command(arm, p.SEEDS[0])
    parsed = []
    monkeypatch.setattr(sys, "argv", ["train.py", *command[command.index("jax_experiments.train") + 1:]])
    monkeypatch.setattr(train, "train", parsed.append)
    train.main()
    config = parsed[0]
    assert all(getattr(config, key) == value for key, value in p.settings(arm).items())
    assert config.seed == p.SEEDS[0]
    assert p.budget(arm) == {"next_iteration": 250, "total_steps": 1010000, "update_count": 1000000}
    assert p.actor_updates(arm) == 500000
    metadata = dict(identity=p.identity(arm, config.seed), runtime_versions=p.RUNTIME,
                    norm_definition=p.NORM_DEFINITION, budget=p.budget(arm),
                    actor_update_count=500000, config=vars(config))
    p.validate(metadata, arm, config.seed)
    metadata["config"]["resac_reward_transform"] = "other"
    with pytest.raises(ValueError, match="controls"):
        p.validate(metadata, arm, config.seed)


def test_only_registered_factors_change_and_initial_weights_match():
    profiles = [{key: value for key, value in p.settings(arm).items()
                 if key not in {"resac_reward_transform", "weight_reg"}} for arm in p.ARMS]
    assert all(profile == profiles[0] for profile in profiles)
    agents = [TorchEquationRESAC(3, 2, Config(algo="resac_torch", hidden_dim=8, ensemble_size=3,
              resac_reward_transform=transform), seed=43) for transform in ("raw", "bus_standardized")]
    for name in ("policy", "critic", "target_critic"):
        for left, right in zip(jax.tree.leaves(nnx.state(getattr(agents[0], name), nnx.Param)),
                               jax.tree.leaves(nnx.state(getattr(agents[1], name), nnx.Param))):
            np.testing.assert_array_equal(left, right)
    assert p.settings("bus_reward_full")["weight_reg"] == p.settings("raw_full")["weight_reg"] == 0.01
    assert Config().resac_reward_transform == "raw"


def test_seed_level_reward_contrasts_and_interaction():
    values = dict(raw_full=-10., raw_no_shift=100., bus_reward_full=90., bus_reward_no_shift=120.)
    audits = [{"protocol": p.PROTOCOL, "seed": seed, "runtime_versions": p.RUNTIME,
        "budgets": {arm: p.budget(arm) for arm in p.ARMS},
        "actor_updates": {arm: p.actor_updates(arm) for arm in p.ARMS},
        "switching": {"event": {arm: {"returns": [value + i, value + i]} for arm, value in values.items()}},
        "stationary": {"event": {arm: {"0": {"return_mean": value + i}} for arm, value in values.items()}},
        "diagnostics": {}} for i, seed in enumerate(p.SEEDS)]
    result = run.summarize(audits)
    assert result["contrasts"]["reward_gain_full"]["mean"] == 100
    assert result["contrasts"]["reward_gain_zero"]["mean"] == 20
    assert result["contrasts"]["reward_by_shift_interaction"]["ci95"] == [80., 80.]
    assert result["stationary"]["bus_reward_full"] == [90., 91., 92.]
    audits[0]["budgets"]["raw_full"]["update_count"] -= 1
    with pytest.raises(ValueError, match="mixed/incomplete"):
        run.summarize(audits)


def test_scheduler_smoke_actor_gates_and_no_checkpoint_sync():
    rows = submit.candidates()
    assert len(rows) == 17 and sum(row["vram"] > 0 for row, _ in rows) == 13
    for row, output in rows:
        assert row["signature"].startswith(submit.PREFIX)
        assert submit.MODULE in row["cmd"]
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        if "train-" in row["signature"]:
            assert row["vram"] == 1800
            assert row["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert row["resume_managed_by_cmd"] and not row["allow_initial_resume_scan_error"]
        if "audit-" in row["signature"]:
            assert row["vram"] == 0 and len(row["wait_for_files"]) == 6
            assert all("checkpoints" not in path for path in row["stage_input_paths"])
    assert len(rows[-1][0]["wait_for_files"]) == 4
