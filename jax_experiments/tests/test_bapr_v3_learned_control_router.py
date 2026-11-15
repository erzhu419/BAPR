from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import bapr_v3_learned_control_router as router


def _load_submit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_learned_control_router.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_learned_control_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_control_equivalence_marginalizes_mode_one_and_two():
    probabilities, controllers = router.aggregate_controller_probabilities(
        np.asarray([0.1, 0.3, 0.4, 0.2]), (0, 2, 2, 3))
    assert controllers == (0, 2, 3)
    np.testing.assert_allclose(probabilities, [0.1, 0.7, 0.2])


def test_router_falls_back_until_history_and_confidence_are_sufficient():
    config = router.RouterConfig(
        hazard_rate=0.002,
        evidence_scale=1.0,
        confidence_threshold=0.7,
        margin_threshold=0.1,
        min_history=8,
    )
    confident = np.asarray([0.92, 0.03, 0.03, 0.02])
    selected, diagnostics = router.select_controller(
        confident, 7, (0, 2, 2, 3), config)
    assert selected == -1
    assert diagnostics["fallback"] == 1.0

    selected, diagnostics = router.select_controller(
        confident, 8, (0, 2, 2, 3), config)
    assert selected == 0
    assert diagnostics["fallback"] == 0.0

    ambiguous = np.asarray([0.45, 0.25, 0.20, 0.10])
    selected, _ = router.select_controller(
        ambiguous, 100, (0, 2, 2, 3), config)
    assert selected == -1


def test_causal_route_does_not_use_current_transition_evidence():
    config = router.RouterConfig(
        hazard_rate=0.01,
        evidence_scale=2.0,
        confidence_threshold=0.6,
        margin_threshold=0.05,
        min_history=1,
        hysteresis_margin=0.0,
    )
    evidence = np.asarray([
        [8.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 8.0],
        [0.0, 0.0, 0.0, 8.0],
    ])
    decisions, posteriors, _ = router.causal_route_trace(
        evidence, (0, 2, 2, 3), config)
    assert decisions[0] == -1
    assert decisions[1] == 0
    assert int(np.argmax(posteriors[1])) == 3
    assert decisions[2] == 3


def test_routing_metrics_use_control_labels_not_physical_identity():
    true_modes = np.asarray([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40)
    expected = np.asarray([0] * 40 + [2] * 80 + [3] * 40)
    metrics = router.routing_metrics(
        expected, true_modes, (0, 2, 2, 3), burnin=0)
    assert metrics["coverage"] == 1.0
    assert metrics["conditional_accuracy"] == 1.0
    assert metrics["wrong_route_rate"] == 0.0
    assert metrics["median_switch_delay"] == 0.0


def test_parameter_npz_roundtrip_uses_template_treedef(tmp_path):
    state = {
        "a": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "b": (jnp.asarray([4, 5], dtype=jnp.int32),),
    }
    path = tmp_path / "params.npz"
    metadata = router.save_parameter_state(path, state)
    template = jax.tree.map(jnp.zeros_like, state)
    restored = router.load_parameter_state(path, template, metadata)
    for actual, expected in zip(
            jax.tree.leaves(restored), jax.tree.leaves(state)):
        np.testing.assert_array_equal(actual, expected)


def test_router_protocol_splits_are_disjoint_and_scheduler_only():
    assert not (
        set(router.TRAIN_EVENT_SEEDS)
        & set(router.VALIDATION_EVENT_SEEDS)
    )
    assert not (
        set(router.TRAIN_EVENT_SEEDS)
        & set(router.HOLDOUT_EVENT_SEEDS)
    )
    assert not (
        set(router.VALIDATION_EVENT_SEEDS)
        & set(router.HOLDOUT_EVENT_SEEDS)
    )

    module = _load_submit_module()
    _, spec, output = module.train_spec("high")
    assert output == router.MANIFEST_PATH
    assert spec["require_node"] == "jtl311linux"
    assert spec["ckpt_glob"] == "train_state.json"
    assert "--resume" in spec["cmd"]
    assert "slurm" not in spec["cmd"].lower()
    assert "auto-adopt" not in spec["cmd"].lower()
