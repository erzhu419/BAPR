"""Structural and semantic tests for persistent joint-damping headroom."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import regime_damping_headroom as protocol
from jax_experiments.analysis import (
    run_regime_damping_headroom_controller as controller,
)
from jax_experiments.analysis import (
    run_regime_damping_headroom_audit as audit,
)
from jax_experiments.envs.persistent_damping_mode_env import (
    ACTION_NOISE_STD,
    FAMILY,
    PersistentDampingModeEnv,
)
from scripts import submit_regime_damping_headroom as submit


def test_splits_and_scheduler_graph_are_registered_before_training():
    assert len(protocol.TRAINING_SEEDS) == 3
    assert len(protocol.AUDIT_EVENT_SEEDS) == 3
    assert not set(protocol.TRAINING_SEEDS) & set(protocol.AUDIT_EVENT_SEEDS)
    assert protocol.FAMILY == FAMILY
    assert protocol.MIN_PASSING_ENVS == 3
    rows = submit.candidates("all", "high")
    assert len(rows) == 49
    assert len({signature for signature, _, _ in rows}) == 49
    training = [spec for signature, spec, _ in rows if "/train/" in signature]
    audits = [spec for signature, spec, _ in rows if "/audit/" in signature]
    assert len(training) == len(audits) == 24
    assert all(spec["vram"] == 2300 for spec in training)
    assert all(spec["allow_gpu_over_one_third"] for spec in training)
    assert all(spec["allowed_nodes"] == submit.GPU_NODES for spec in training)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in audits)


def test_wrapper_commands_install_isolated_environment_entries():
    command = controller.training_command(
        protocol.ENVS[0], "robust", protocol.TRAINING_SEEDS[0])
    assert "jax_experiments.analysis.train_regime_damping_headroom_entry" \
        in command
    evaluation = audit.evaluation_command(
        protocol.ENVS[0], "robust", protocol.TRAINING_SEEDS[0],
        protocol.AUDIT_EVENT_SEEDS[0], protocol.AUDIT_ROOT / "test")
    assert "jax_experiments.analysis.final_task_sweep_regime_damping" \
        in evaluation


def test_damping_modes_are_equal_norm_persistent_physics():
    env = PersistentDampingModeEnv(
        "HalfCheetah-v2", dwell_steps=5, seed=17, backend="spring")
    tasks = env.sample_tasks(4)
    base = np.asarray(env.base_sys.dof.damping)
    norms = [float(task["damping_log_l2"]) for task in tasks]
    assert np.allclose(norms, norms[0], rtol=1e-6, atol=1e-6)
    assert all(float(task["action_noise_std"]) == ACTION_NOISE_STD
               for task in tasks)
    assert all(np.array_equal(task["gravity"], tasks[0]["gravity"])
               for task in tasks)
    assert all(not np.array_equal(task["dof_damping"], base)
               for task in tasks)
    assert len({tuple(task["damping_multiplier"]) for task in tasks}) == 4
    for mode, task in enumerate(tasks):
        assert np.allclose(
            np.asarray(env._mode_sys[mode].dof.damping),
            np.asarray(task["dof_damping"]),
        )
    before = np.asarray(tasks[0]["dof_damping"]).copy()
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[0])
    env.reset()
    env.step(np.zeros((env.act_dim,), dtype=np.float32))
    assert env.current_task_id == 0
    assert np.array_equal(before, env.sample_tasks(4)[0]["dof_damping"])


if __name__ == "__main__":
    test_splits_and_scheduler_graph_are_registered_before_training()
    test_wrapper_commands_install_isolated_environment_entries()
    test_damping_modes_are_equal_norm_persistent_physics()
