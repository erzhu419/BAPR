"""Tests for the nonlinear posterior-conditioned residual protocol."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_posterior_residual_policy as protocol,
)
from jax_experiments.analysis import (
    train_bapr_v3_posterior_residual_policy as trainer,
)
from jax_experiments.analysis import (
    run_bapr_v3_posterior_residual_policy_audit as audit,
)


def load_submit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_posterior_residual_policy.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_posterior_residual_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_posterior_strength_is_zero_when_robust_has_best_utility():
    matrix = jnp.asarray([
        [10.0, 20.0],
        [30.0, 0.0],
        [10.0, 20.0],
        [10.0, 0.0],
    ])
    strength = protocol.posterior_strength(
        jnp.full((4,), 0.25), matrix, 0, (1,), 10.0)
    assert float(strength) == 0.0


def test_policy_is_exactly_robust_when_strength_is_zero():
    policy = protocol.make_policy(obs_dim=3, act_dim=2, seed=0)
    robust = jnp.asarray([0.25, -0.5])
    action = policy(
        jnp.zeros((3,)), jnp.full((4,), 0.25), robust,
        jnp.asarray(0.0), jnp.zeros((3,)), jnp.ones((3,)))
    assert np.array_equal(np.asarray(action), np.asarray(robust))


def test_model_parameter_layout_is_seed_invariant():
    left = nnx.state(protocol.make_policy(3, 2, 0), nnx.Param)
    right = nnx.state(protocol.make_policy(3, 2, 1), nnx.Param)
    assert len(jax.tree.leaves(left)) == len(jax.tree.leaves(right))
    assert [x.shape for x in jax.tree.leaves(left)] \
        == [x.shape for x in jax.tree.leaves(right)]


def test_one_jitted_training_update_is_finite():
    model = protocol.make_policy(3, 2, 0)
    utility = (
        np.asarray([
            [10.0, 20.0], [30.0, 0.0],
            [10.0, 20.0], [10.0, 0.0],
        ], dtype=np.float32),
        0,
        (1,),
        10.0,
    )
    optimizer, update, _ = trainer.build_optimizer(model, utility)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    batch = {
        "obs": jnp.zeros((8, 3)),
        "posterior": jnp.tile(jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
                              (8, 1)),
        "robust_action": jnp.zeros((8, 2)),
        "teacher_action": jnp.full((8, 2), 0.5),
    }
    next_params, _, metrics = update(
        params, opt_state, batch, jnp.zeros((3,)), jnp.ones((3,)))
    assert all(np.all(np.isfinite(np.asarray(value)))
               for value in jax.tree.leaves(next_params))
    assert np.all(np.isfinite(np.asarray(metrics)))


def test_training_selection_and_return_seeds_are_disjoint():
    used = set(protocol.TRAIN_EVENT_SEEDS) \
        | set(protocol.MODEL_SELECTION_EVENT_SEEDS) \
        | set(protocol.DEVELOPMENT_RETURN_EVENT_SEEDS)
    assert len(used) == (
        len(protocol.TRAIN_EVENT_SEEDS)
        + len(protocol.MODEL_SELECTION_EVENT_SEEDS)
        + len(protocol.DEVELOPMENT_RETURN_EVENT_SEEDS))
    assert not used.intersection(protocol.SEALED_CONFIRMATION_EVENT_SEEDS)


def test_audit_accepts_float32_utility_roundoff_only():
    assert audit.residual_strengths_agree(
        0.857052862644, 0.857051896441)
    assert not audit.residual_strengths_agree(0.857, 0.856)


def test_submission_is_scheduler_only_and_pinned_to_artifacts():
    module = load_submit_module()
    signature, task, output = module.training_spec("high")
    assert signature.endswith("/train")
    assert output == protocol.MANIFEST_PATH
    assert task["require_node"] == "jtl311linux"
    assert task["ckpt_glob"] == "router_manifest.json"
    assert "--resume" in task["cmd"]
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()
