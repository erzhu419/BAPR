from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from jax_experiments.analysis import bapr_v3_sequence_router as protocol
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility
from jax_experiments.analysis import train_bapr_v3_sequence_router as trainer
from jax_experiments.analysis import diagnose_bapr_v3_sequence_router as diagnostic
from jax_experiments.analysis import train_bapr_v3_sequence_router_v2 as curriculum
from jax_experiments.analysis import train_bapr_v3_sequence_router_dual as dual
from jax_experiments.analysis import train_bapr_v3_sequence_router_gate as gate


def _load_submit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_sequence_router.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_sequence_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sequence_filter_is_causal_and_normalized():
    model = trainer.make_sequence_model(seed=0)
    prefix = jnp.asarray(np.random.default_rng(0).normal(size=(12, 4)))
    first = jnp.concatenate([prefix, jnp.zeros((8, 4))], axis=0)
    second = jnp.concatenate([prefix, jnp.ones((8, 4)) * 20.0], axis=0)
    _, first_logits = model.sequence(first)
    _, second_logits = model.sequence(second)
    np.testing.assert_allclose(first_logits[:12], second_logits[:12])
    posterior = np.array(jnp.exp(first_logits - jnp.max(
        first_logits, axis=-1, keepdims=True)), copy=True)
    posterior /= posterior.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0)


def test_sequence_batch_preserves_shapes_and_switch_weights():
    sequence = {
        "evidence": np.zeros((256, 4), dtype=np.float32),
        "labels": np.asarray([0] * 128 + [1] * 128, dtype=np.int32),
        "weights": np.asarray([1.0] * 128 + [3.0] * 64 + [1.0] * 64),
        "identity": {"kind": "full_cycle"},
    }
    batch = trainer.sample_batch(
        [sequence], 4, 64, np.random.default_rng(0))
    assert batch[0].shape == (4, 64, 4)
    assert batch[1].shape == (4, 64)
    assert batch[2].shape == (4, 64)
    assert set(np.asarray(batch[2]).ravel()).issubset({1.0, 3.0})


def test_sequence_training_update_keeps_gru_rng_state():
    model = trainer.make_sequence_model(seed=0)
    update, optimizer_state = trainer.build_update(
        model, learning_rate=3e-4, burnin=2)
    params = trainer.nnx.state(model, trainer.nnx.Param)
    evidence = jnp.zeros((2, 8, 4), dtype=jnp.float32)
    labels = jnp.zeros((2, 8), dtype=jnp.int32)
    weights = jnp.ones((2, 8), dtype=jnp.float32)

    next_params, _, loss, accuracy = update(
        params, optimizer_state, evidence, labels, weights)

    assert set(next_params) == set(params)
    assert np.isfinite(float(loss))
    assert 0.0 <= float(accuracy) <= 1.0


def test_sequence_diagnostic_decision_trace_respects_causal_fallback():
    table = {
        "nondominated_controllers": [4, 0, 2, 3],
        "rows": {
            str(mode): {
                "mean_returns": {
                    str(controller): float(controller == expected) * 10.0
                    for controller in (4, 0, 2, 3)
                }
            }
            for mode, expected in enumerate((0, 4, 2, 3))
        },
    }
    base = protocol.emission.RouterConfig(
        hazard_rate=0.005,
        evidence_scale=0.25,
        confidence_threshold=0.8,
        margin_threshold=0.02,
        min_history=2,
    )
    posterior = np.asarray([
        [0.99, 0.005, 0.003, 0.002],
        [0.99, 0.005, 0.003, 0.002],
        [0.99, 0.005, 0.003, 0.002],
    ])
    decisions = diagnostic.decision_trace(posterior, table, base)
    assert decisions.tolist() == [-1, -1, 0]
    no_history = replace(base, min_history=0)
    assert diagnostic.decision_trace(
        posterior, table, no_history).tolist() == [-1, 0, 0]


def test_switch_centered_batch_places_real_switch_inside_context():
    evidence = np.zeros((2, 1000, 4), dtype=np.float32)
    labels = np.stack([
        np.zeros((1000,), dtype=np.int32),
        np.repeat(np.arange(4), 250).astype(np.int32),
    ])
    kind = np.asarray([0, 1], dtype=np.int8)
    ages = curriculum.switch_age(labels)
    variant = {
        "context_length": 128,
        "switch_fraction": 1.0,
        "switch_weight": 8.0,
        "switch_span": 32,
    }
    batch = curriculum.sample_curriculum_batch(
        evidence, labels, kind, ages, variant,
        np.random.default_rng(0), batch_size=8)
    batch_labels = np.asarray(batch[1])
    batch_weights = np.asarray(batch[2])
    assert np.all(np.any(batch_labels[:, 1:] != batch_labels[:, :-1], axis=1))
    assert np.all(np.any(batch_weights == 8.0, axis=1))
    assert not np.any(batch_weights > 8.0)


def test_dual_window_gate_uses_fast_only_after_real_switch():
    labels = np.asarray([
        [0] * 1000,
        [0] * 250 + [1] * 250 + [2] * 250 + [3] * 250,
    ])
    kind = np.asarray([0, 1], dtype=np.int8)
    oracle_map = [0, 4, 2, 3]
    slow_logits = np.full((2, 1000, 4), -10.0, dtype=np.float32)
    fast_logits = np.full_like(slow_logits, -10.0)
    for row in range(2):
        slow_logits[row, np.arange(1000), labels[row]] = 10.0
        fast_logits[row, np.arange(1000), labels[row]] = 10.0
    # Causal action predictions use the previous row. Make the slow expert lag
    # for 16 observations and let the fast expert recover after 4.
    for point in (250, 500, 750):
        previous = int(labels[1, point - 1])
        slow_logits[1, point:point + 16] = -10.0
        slow_logits[1, point:point + 16, previous] = 10.0
        fast_logits[1, point:point + 4] = -10.0
        fast_logits[1, point:point + 4, previous] = 10.0
    slow_only = dual.window_gate_metrics(
        slow_logits, fast_logits, labels, kind, oracle_map, window=0)
    gated = dual.window_gate_metrics(
        slow_logits, fast_logits, labels, kind, oracle_map, window=16)
    assert gated["full_cycle"]["action_accuracy"] \
        > slow_only["full_cycle"]["action_accuracy"]
    assert gated["stationary"]["action_accuracy"] == 1.0


def test_causal_gate_features_use_only_prefix_logits():
    rng = np.random.default_rng(0)
    slow_prefix = rng.normal(size=(2, 12, 4)).astype(np.float32)
    fast_prefix = rng.normal(size=(2, 12, 4)).astype(np.float32)
    slow_a = np.concatenate([slow_prefix, np.zeros((2, 8, 4))], axis=1)
    slow_b = np.concatenate([slow_prefix, np.ones((2, 8, 4)) * 20], axis=1)
    fast_a = np.concatenate([fast_prefix, np.zeros((2, 8, 4))], axis=1)
    fast_b = np.concatenate([fast_prefix, np.ones((2, 8, 4)) * -20], axis=1)
    features_a, _, _ = gate.gate_features(slow_a, fast_a, [0, 4, 2, 3])
    features_b, _, _ = gate.gate_features(slow_b, fast_b, [0, 4, 2, 3])
    # Action at t uses posterior emitted after t-1, so the first future row can
    # affect only the following feature row.
    np.testing.assert_allclose(features_a[:, :13], features_b[:, :13])


def test_sequence_protocol_splits_are_disjoint():
    prior = {
        *protocol.emission.TRAIN_EVENT_SEEDS,
        *protocol.emission.VALIDATION_EVENT_SEEDS,
        *protocol.emission.HOLDOUT_EVENT_SEEDS,
        *utility.VALIDATION_EVENT_SEEDS,
        *utility.HOLDOUT_EVENT_SEEDS,
    }
    assert not prior.intersection(protocol.TRAIN_EVENT_SEEDS)
    assert not prior.intersection(protocol.VALIDATION_EVENT_SEEDS)
    assert not set(protocol.TRAIN_EVENT_SEEDS).intersection(
        protocol.VALIDATION_EVENT_SEEDS)


def test_sequence_training_submission_is_scheduler_only_and_resumable():
    module = _load_submit_module()
    signature, spec, output = module.train_spec("high")
    assert "sequence-router" in signature
    assert signature.endswith("/v1/train")
    assert output == protocol.MANIFEST_PATH
    assert spec["require_node"] == "jtl311linux"
    assert spec["ckpt_glob"] == "train_state.json"
    assert "--resume" in spec["cmd"]
    assert "slurm" not in spec["cmd"].lower()
    assert "auto-adopt" not in spec["cmd"].lower()


def test_sequence_diagnostic_submission_is_scheduler_only():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_sequence_router_diagnostic.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_sequence_router_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    signature, task, output = module.diagnostic_spec("high")
    assert signature.endswith("/v1/diagnostic")
    assert output == diagnostic.JSON_PATH
    assert task["require_node"] == "jtl311linux"
    assert "7100" not in task["cmd"]
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()


def test_sequence_curriculum_submission_is_scheduler_only_and_resumable():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_sequence_router_v2.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_sequence_router_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    signature, task, output = module.train_spec("high")
    assert signature.endswith("/v2/curriculum")
    assert output == curriculum.protocol.MANIFEST_PATH
    assert task["ckpt_glob"] == "train_state.json"
    assert task["require_node"] == "jtl311linux"
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()


def test_sequence_dual_submission_is_scheduler_only_and_resumable():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_sequence_router_dual.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_sequence_router_dual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    signature, task, output = module.train_spec("high")
    assert signature.endswith("/v3/dual-oracle")
    assert output == dual.protocol.MANIFEST_PATH
    assert task["ckpt_glob"] == "train_state.json"
    assert task["require_node"] == "jtl311linux"
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()


def test_sequence_gate_submission_is_scheduler_only_and_resumable():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_sequence_router_gate.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_sequence_router_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    signature, task, output = module.train_spec("high")
    assert signature.endswith("/v4/causal-gate")
    assert output == gate.protocol.MANIFEST_PATH
    assert task["ckpt_glob"] == "train_state.json"
    assert task["require_node"] == "jtl311linux"
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()
