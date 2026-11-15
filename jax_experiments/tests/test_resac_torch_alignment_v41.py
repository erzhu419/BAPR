"""Real update, checkpoint continuation and frozen experimental contrasts."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import pytest

from jax_experiments.algos.resac_torch import TorchEquationRESAC, sample_std
from jax_experiments.analysis import resac_torch_alignment_protocol_v41 as p
from jax_experiments.analysis import run_resac_torch_alignment_v41 as run
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from scripts import submit_resac_torch_alignment_v41 as submit


def config(reward_transform="raw"):
    return Config(algo="resac_torch", hidden_dim=8, ensemble_size=3,
                  resac_clip_norm=1.0, resac_critic_actor_ratio=2, resac_beta_bc=0.001,
                  resac_reward_transform=reward_transform)


def batch(count):
    rng = np.random.default_rng(17)
    return {name: jnp.asarray(rng.normal(size=(count, 6, size)), dtype=jnp.float32)
            for name, size in (("obs", 3), ("act", 2), ("rew", 1), ("next_obs", 3))} | {
            "done": jnp.zeros((count, 6, 1))}


def test_torch_initialization_and_std():
    agent = TorchEquationRESAC(3, 2, config(), seed=37)
    assert float(agent.alpha[0]) == 1.0
    for layer in agent.policy.layers:
        np.testing.assert_allclose(layer.bias.value, 0.1)
        assert abs(np.asarray(layer.kernel.value)).max() <= 1 / np.sqrt(layer.kernel.value.shape[1])
    for head in (agent.policy.mean_head, agent.policy.log_std_head):
        assert abs(np.asarray(head.kernel.value)).max() <= 0.001
    for layer in agent.critic.layers:
        bound = 1 / np.sqrt(layer.kernel.value.shape[-1])
        assert abs(np.asarray(layer.kernel.value)).max() <= bound
        assert abs(np.asarray(layer.bias.value)).max() <= bound
    assert agent.critic.norm.epsilon == 1e-5
    np.testing.assert_allclose(sample_std(jnp.array([[1., 3.], [2., 5.], [4., 7.]])),
                               np.std([[1., 3.], [2., 5.], [4., 7.]], axis=0, ddof=1), rtol=1e-6)
    np.testing.assert_array_equal(jax.grad(lambda x: sample_std(x).sum())(jnp.ones((3, 4))),
                                  np.zeros((3, 4)))


def test_treatments_share_initial_weights_and_only_disable_shift():
    positive, zero = config(), config()
    zero.weight_reg = 0.0
    agents = [TorchEquationRESAC(3, 2, settings, seed=41) for settings in (positive, zero)]
    for name in ("policy", "critic", "target_critic"):
        for left, right in zip(jax.tree.leaves(nnx.state(getattr(agents[0], name), nnx.Param)),
                               jax.tree.leaves(nnx.state(getattr(agents[1], name), nnx.Param))):
            np.testing.assert_array_equal(left, right)
    for settings, agent in zip((positive, zero), agents):
        metrics = agent.multi_update(batch(3))
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics["actor_update_rate"] == pytest.approx(2 / 3)
        assert (metrics["reg_bonus_mean"] > 0) == (settings.weight_reg > 0)


@pytest.mark.parametrize("reward_transform", ("raw", "bus_standardized"))
def test_checkpoint_preserves_rng_adam_and_odd_actor_phase(tmp_path, reward_transform):
    agent = TorchEquationRESAC(3, 2, config(reward_transform), seed=43)
    agent.multi_update(batch(3))
    replay = ReplayBuffer(3, 2, capacity=12)
    replay.push_batch(np.ones((6, 3)), np.zeros((6, 2)), np.ones((6, 1)),
                      np.ones((6, 3)), np.zeros((6, 1)))
    logger = Logger(str(tmp_path / "logs"))
    save_checkpoint(str(tmp_path / "checkpoints"), agent, replay, logger, 2, 12, "resac_torch")
    restored = TorchEquationRESAC(3, 2, config(reward_transform), seed=47)
    iteration, steps = load_checkpoint(str(tmp_path / "checkpoints"), restored,
        ReplayBuffer(3, 2, capacity=12), Logger(str(tmp_path / "restored")), "resac_torch")
    assert (iteration, steps, restored.update_count) == (3, 12, 3)
    expected, actual = agent.multi_update(batch(1)), restored.multi_update(batch(1))
    assert expected["actor_update_rate"] == actual["actor_update_rate"] == 0
    expected, actual = agent.multi_update(batch(1)), restored.multi_update(batch(1))
    assert expected["actor_update_rate"] == actual["actor_update_rate"] == 1
    for name in ("policy", "critic", "target_critic"):
        for left, right in zip(jax.tree.leaves(nnx.state(getattr(agent, name), nnx.Param)),
                               jax.tree.leaves(nnx.state(getattr(restored, name), nnx.Param))):
            np.testing.assert_array_equal(left, right)
    for name in ("policy_opt_state", "critic_opt_state", "alpha_opt_state"):
        for left, right in zip(jax.tree.leaves(getattr(agent, name)), jax.tree.leaves(getattr(restored, name))):
            np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(agent.log_alpha, restored.log_alpha)
    assert agent.update_count == restored.update_count == 5


def test_aligned_commands_only_differ_in_norm_weight():
    commands = [run.training_command(arm, p.SEEDS[0]) for arm in ("aligned_full", "aligned_no_shift")]
    ignore = {"--weight_reg", "--save_root", "--run_name"}
    normalized = [[value for i, value in enumerate(command) if value not in ignore
                   and not (i and command[i - 1] in ignore)] for command in commands]
    assert normalized[0] == normalized[1]
    for command in commands:
        for name, value in (("algo", "resac_torch"), ("ensemble_size", "10"),
                            ("resac_critic_actor_ratio", "2"), ("resac_clip_norm", "1.0"), ("alpha", "1.0")):
            assert command[command.index("--" + name) + 1] == value
    assert p.actor_updates("fork") == 525000 and p.actor_updates("aligned_full") == 262500
    metadata = dict(identity=p.identity("aligned_full", p.SEEDS[0]), budget=p.budget(),
                    actor_update_count=262500, runtime_versions=p.RUNTIME, norm_definition=p.NORM_DEFINITION,
                    config={**vars(Config()), **p.settings("aligned_full"), "seed": p.SEEDS[0]})
    p.validate(metadata, "aligned_full", p.SEEDS[0])
    metadata["actor_update_count"] += 1
    with pytest.raises(ValueError, match="budget"):
        p.validate(metadata, "aligned_full", p.SEEDS[0])


@pytest.mark.parametrize("arm", p.ARMS)
def test_full_training_cli_preserves_profile(arm, monkeypatch):
    import sys
    from jax_experiments import train

    command = run.training_command(arm, p.SEEDS[0])
    configurations = []
    monkeypatch.setattr(sys, "argv", ["train.py", *command[command.index("jax_experiments.train") + 1:]])
    monkeypatch.setattr(train, "train", configurations.append)
    train.main()
    config = configurations[0]
    assert config.seed == p.SEEDS[0]
    assert all(getattr(config, key) == value for key, value in p.settings(arm).items())


def test_scheduler_gates_and_compact_sync():
    rows = submit.candidates()
    assert len(rows) == 14 and sum(row["vram"] > 0 for row, output in rows) == 10
    assert str(p.REFERENCE.parent) in rows[0][0]["stage_input_paths"]
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        if "train-" in row["signature"]:
            assert row["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert row["resume_managed_by_cmd"]
        if "audit-" in row["signature"]:
            assert row["vram"] == 0 and len(row["wait_for_files"]) == 5
    assert len(rows[-1][0]["wait_for_files"]) == 4
