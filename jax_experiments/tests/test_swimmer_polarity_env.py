import jax
import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest
from flax import nnx

from jax_experiments.common.checkpoint import _to_numpy_tree
from jax_experiments.configs.default import Config
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.train import make_algo


def test_swimmer_step_matches_native_brax_reward_and_observation():
    env = StochasticModeEnv(
        "Swimmer-v2", family="actuator_polarity_2d",
        backend="generalized", seed=17,
    )
    try:
        state = env.env.reset(jax.random.PRNGKey(19))
        action = np.asarray([0.2, -0.1], dtype=np.float32)
        native = env.env.step(state, action)
        custom = env._step_fn(env.base_sys, state, action)
        np.testing.assert_allclose(custom.obs, native.obs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom.reward, native.reward, rtol=1e-5, atol=1e-5)
        assert float(custom.done) == float(native.done) == 0.0
    finally:
        env.close()


def test_swimmer_modes_are_distinct_invertible_and_switch_causally():
    env = StochasticModeEnv(
        "Swimmer-v2", family="actuator_polarity_2d",
        backend="generalized", dwell_steps=2, seed=23,
    )
    try:
        tasks = env.sample_tasks(4)
        gains = [tuple(np.asarray(task["action_gain"]).tolist()) for task in tasks]
        assert gains == [
            (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)
        ]
        env.configure_eval_mode_sequence(tasks, [0, 1, 2, 3], 2)
        env.reset()
        used = []
        for _ in range(8):
            _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
            assert not done
            used.append(info["mode_used"])
        assert used == [0, 0, 1, 1, 2, 2, 3, 3]
    finally:
        env.close()


def test_two_dimensional_polarity_family_rejects_other_action_sizes():
    with pytest.raises(ValueError, match="exactly two actuators"):
        StochasticModeEnv(
            "HalfCheetah-v2", family="actuator_polarity_2d",
            backend="spring", seed=1,
        )


def test_screen_training_and_scheduler_preserve_budget_and_small_results():
    from jax_experiments.analysis import swimmer_polarity_screen_v1 as screen
    from scripts import submit_swimmer_polarity_screen_v1 as submit

    for arm in screen.ARMS:
        command = screen.train_command(arm, screen.SEEDS[0])
        assert command[command.index("--backend") + 1] == "generalized"
        assert command[command.index("--max_iters") + 1] == "2100"
        assert command[command.index("--samples_per_iter") + 1] == "4000"
        assert command[command.index("--updates_per_iter") + 1] == "250"
        assert ("--stochastic_mode_fixed_id" in command) == (arm == "reference")

    rows = submit.candidates("normal")
    assert len(rows) == 10
    assert sum(spec["vram"] > 0 for _, spec, _ in rows) == 6
    for _, spec, _ in rows:
        assert "local" not in spec["allowed_nodes"]
        if spec["vram"]:
            assert spec["result_dir"] == spec["local_result_dir"]
            assert "eval_bundles_swimmer" in spec["result_dir"]
            assert "checkpoints" in spec["ckpt_dir"]
        else:
            assert "results_swimmer_polarity_screen_v1/robust" not in spec["result_dir"]
    for seed in screen.SEEDS:
        audit_spec = next(spec for signature, spec, _ in rows
                          if signature.endswith(f"/audit/seed-{seed}"))
        assert len(audit_spec["wait_for_files"]) == 2
        assert all(path.endswith("bundle.json") for path in audit_spec["wait_for_files"])


def test_screen_actor_loads_compact_policy_bundle(tmp_path, monkeypatch):
    from jax_experiments.analysis import swimmer_polarity_screen_v1 as screen

    monkeypatch.setattr(screen, "BUNDLE_ROOT", tmp_path)
    seed = screen.SEEDS[0]
    config = Config()
    config.algo = "sac"
    config.env_name = "Swimmer-v2"
    config.brax_backend = "generalized"
    config.seed = seed
    config.ensemble_size = 2
    agent = make_algo("sac", 8, 2, config)
    directory = screen.bundle_dir("robust", seed)
    (directory / "logs").mkdir(parents=True)
    (directory / "bundle.json").write_text(json.dumps({
        "arm": "robust", "seed": seed, "total_steps": screen.TOTAL_STEPS,
    }))
    (directory / "logs/protocol_signature.json").write_text(json.dumps({
        "config": asdict(config),
    }))
    with (directory / "policy_params.pkl").open("wb") as handle:
        pickle.dump(_to_numpy_tree(nnx.state(agent.policy, nnx.Param)), handle)
    action = screen.load_actor("robust", seed)(np.zeros(8, dtype=np.float32))
    assert action.shape == (2,)
    assert np.isfinite(action).all()
