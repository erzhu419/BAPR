"""Protocol tests for median-ensemble policy distillation."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation as protocol,
)
from jax_experiments.analysis.run_regime_polarity_policy_distillation_audit import (
    arm_labels,
)
from jax_experiments.analysis.train_regime_polarity_policy_distillation import (
    _finalize_rows,
)
from scripts import submit_regime_polarity_policy_distillation as submitter


def test_distillation_splits_are_disjoint_and_retrospective():
    splits = (
        protocol.TRAIN_EVENT_SEEDS,
        protocol.DAGGER_EVENT_SEEDS,
        protocol.VALIDATION_EVENT_SEEDS,
        protocol.AUDIT_EVENT_SEEDS,
    )
    flattened = [seed for split in splits for seed in split]
    assert len(flattened) == len(set(flattened))
    assert set(flattened).isdisjoint(protocol.ensemble.EVENT_SEEDS)
    identity = protocol.model_identity("combined", protocol.STUDENT_SEEDS[0])
    assert identity["development_only"] is True
    assert identity["reduction"] == "median"
    assert identity["student_online_inputs"] == [
        "observation", "soft_mode_posterior"]


def test_teacher_groups_keep_controller_provenance():
    development = protocol.controller_keys("development")
    final = protocol.controller_keys("final")
    combined = protocol.controller_keys("combined")
    assert len(development) == 5
    assert len(final) == 5
    assert combined == development + final
    assert set(development).isdisjoint(final)


def test_audit_uses_individual_robust_baselines_only():
    for group in protocol.TEACHER_GROUPS:
        labels = arm_labels(group)
        robust = [label for label in labels if label.startswith("robust_")]
        assert len(robust) == len(protocol.controller_keys(group))
        assert "robust_mean" not in labels
        assert "robust_median" not in labels
        assert labels[-3:] == (
            "teacher_oracle_median",
            "teacher_learned_median",
            "student_learned",
        )


def test_dataset_validation_rejects_nonfinite_rows():
    valid = {
        "obs": [np.zeros((3,), dtype=np.float32)],
        "context": [np.full((4,), 0.25, dtype=np.float32)],
        "target_action": [np.zeros((2,), dtype=np.float32)],
        "mode": [0],
    }
    payload = _finalize_rows(valid)
    assert payload["obs"].shape == (1, 3)
    broken = {key: list(value) for key, value in valid.items()}
    broken["target_action"] = [np.asarray([np.nan, 0.0])]
    try:
        _finalize_rows(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite distillation target was accepted")


def test_scheduler_graph_is_gpu_train_then_cpu_audit():
    train = submitter.candidates("train", "high")
    audits = submitter.candidates("audit", "high")
    analyses = submitter.candidates("analysis", "high")
    assert len(train) == 9
    assert len(audits) == 27
    assert len(analyses) == 1
    signatures = [row[0] for row in train + audits + analyses]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in train:
        assert spec["allowed_nodes"] == ["jtl110gpu", "jtl110gpu2"]
        assert spec["vram"] == 1800
        assert spec["cpu"] == 2
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.12" in spec["cmd"]
        assert "node007" not in spec["allowed_nodes"]
    for _, spec, _ in audits:
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
        assert any(path.endswith("model_manifest.json")
                   for path in spec["wait_for_files"])
    assert len(analyses[0][1]["wait_for_files"]) == 27
