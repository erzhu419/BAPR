"""Protocol tests for closed-loop policy compression v2."""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)
from scripts import (
    submit_regime_polarity_policy_distillation_control_v2 as submitter,
)


def _dataset(rows: int = 8):
    return {
        "obs": np.zeros((rows, 17), dtype=np.float32),
        "context": np.full((rows, 4), 0.25, dtype=np.float32),
        "target_action": np.zeros((rows, 6), dtype=np.float32),
        "mode": np.arange(rows, dtype=np.int32) % 4,
        "weight": np.ones((rows,), dtype=np.float32),
    }


def test_development_splits_are_disjoint_from_sealed_confirmation():
    protocol.assert_split_integrity()
    splits = (
        protocol.TRAIN_EVENT_SEEDS,
        protocol.DAGGER_EVENT_SEEDS,
        protocol.SUPERVISED_VALIDATION_EVENT_SEEDS,
        protocol.CONTROL_VALIDATION_EVENT_SEEDS,
        protocol.AUDIT_EVENT_SEEDS,
    )
    flattened = [seed for split in splits for seed in split]
    assert len(flattened) == len(set(flattened))
    assert set(flattened).isdisjoint(protocol.SEALED_CONFIRMATION_EVENT_SEEDS)
    for variant in protocol.VARIANTS:
        identity = protocol.model_identity(variant, protocol.STUDENT_SEEDS[0])
        assert identity["development_only"] is True
        assert identity["teacher_group"] == "combined"
        assert identity["checkpoint_selection"] == (
            "independent_closed_loop_switching_return")


def test_student_architectures_are_finite_and_context_sensitive():
    observations = jnp.tile(
        jnp.linspace(-1.0, 1.0, 17, dtype=jnp.float32)[None], (3, 1))
    contexts = jnp.eye(4, dtype=jnp.float32)[:3]
    for variant in protocol.VARIANTS:
        model = model_lib.make_student(variant, 17, 6, 1709)
        params = nnx.state(model, nnx.Param)
        actions = np.asarray(model_lib.build_student_batch_action(model)(
            params, observations, contexts))
        assert actions.shape == (3, 6)
        assert np.all(np.isfinite(actions))
        assert not np.allclose(actions[0], actions[1])


def test_return_aware_weights_are_normalized_and_nonuniform():
    modes = np.asarray([0] * 50 + [1] * 50, dtype=np.int32)
    target = np.zeros((100, 3), dtype=np.float32)
    student = np.zeros_like(target)
    student[45:60] = 0.5
    weights, metrics = trainer.sample_weights(
        [2000.0, 2100.0, 2050.0],
        [1600.0, 1650.0, 1550.0],
        modes,
        target,
        student,
    )
    assert weights.shape == (100,)
    assert np.isclose(np.mean(weights), 1.0, atol=1e-5)
    assert float(np.max(weights)) > float(np.min(weights))
    assert metrics["normalized_regret"] > 0.0
    assert metrics["student_return_mean"] < metrics["teacher_return_mean"]


def test_weighted_objective_is_finite_for_all_variants():
    batch = trainer._jax_batch(_dataset(16))
    for variant in protocol.VARIANTS:
        model = model_lib.make_student(variant, 17, 6, 1811)
        params = nnx.state(model, nnx.Param)
        _, _, evaluate = trainer.build_optimizer(model)
        loss, metrics = evaluate(params, batch)
        values = np.asarray([loss, *metrics], dtype=np.float64)
        assert np.all(np.isfinite(values))


def test_phase_checkpoint_roundtrip_restores_increment_and_selection():
    original_root = protocol.WORK_ROOT
    with tempfile.TemporaryDirectory() as temporary:
        protocol.WORK_ROOT = Path(temporary)
        try:
            variant = "mode_heads_return"
            seed = protocol.STUDENT_SEEDS[0]
            model = model_lib.make_student(variant, 17, 6, seed)
            params = nnx.state(model, nnx.Param)
            phase_metrics = {
                "phase_index": 0,
                "best_validation_loss": 0.1,
                "control_validation": {"selection_score": 123.0},
            }
            collection = {"kind": "initial", "total_rows": 8}
            trainer._save_phase_checkpoint(
                variant, seed, 0, model, params, params, _dataset(),
                phase_metrics, collection, 0, 123.0)
            restored = trainer._load_phase_checkpoints(variant, seed, model)
            assert restored is not None
            assert restored["latest_phase"] == 0
            assert restored["selected_phase"] == 0
            assert restored["selected_score"] == 123.0
            assert len(restored["dataset"]["obs"]) == 8
        finally:
            protocol.WORK_ROOT = original_root


def test_scheduler_graph_is_nine_gpu_then_twenty_seven_cpu_audits():
    train = submitter.candidates("train", "high")
    audits = submitter.candidates("audit", "high")
    analyses = submitter.candidates("analysis", "high")
    assert len(train) == 9
    assert len(audits) == 27
    assert len(analyses) == 1
    signatures = [row[0] for row in train + audits + analyses]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in train:
        assert spec["vram"] == 2400
        assert spec["cpu"] == 2
        assert spec["allowed_nodes"] == submitter.GPU_NODES
        assert "jtl311linux" in spec["allowed_nodes"]
        assert spec["ckpt_dir"].startswith(str(protocol.WORK_ROOT))
        assert "--resume" in spec["cmd"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
        assert any(path.endswith("model_manifest.json")
                   for path in spec["wait_for_files"])
    assert len(analyses[0][1]["wait_for_files"]) == 27
