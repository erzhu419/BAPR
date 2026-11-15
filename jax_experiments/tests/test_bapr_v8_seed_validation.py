"""Focused invariants for the BAPR-v8 independent-seed protocol."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from jax_experiments.analysis import bapr_v8_seed_validation as protocol
from jax_experiments.analysis.run_bapr_v8_seed_controller import training_command


def _submit_module():
    path = protocol.ROOT / "scripts" / "submit_bapr_v8_seed_validation.py"
    spec = importlib.util.spec_from_file_location("submit_bapr_v8_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_matrix_is_one_seed_per_task():
    submit = _submit_module()
    rows = submit.candidates("training", "high")
    assert len(rows) == 35
    signatures = [signature for signature, _, _ in rows]
    assert len(set(signatures)) == 35
    for seed in protocol.TRAINING_SEEDS:
        assert sum(f"/seed-{seed}/" in signature
                   for signature in signatures) == 7
    for _, task, _ in rows:
        assert task.get("require_node") is None
        assert task.get("allowed_nodes") is None
        assert task["vram"] == 2048
        assert task["resume_managed_by_cmd"] is True
        assert task["allow_initial_resume_scan_error"] is False


def test_dependency_matrix_waits_for_training_artifacts():
    submit = _submit_module()
    calibrations = submit.candidates("calibration", "high")
    audits = submit.candidates("audit", "high")
    assert len(calibrations) == 5
    assert len(audits) == 25
    for seed, (_, task, _) in zip(protocol.TRAINING_SEEDS, calibrations):
        assert set(task["wait_for_files"]) == {
            str(path) for path in protocol.bundle_dependencies_for_seed(seed)}
        assert len(task["wait_for_files"]) == 28
    for _, task, _ in audits:
        assert task["vram"] == 0
        assert set(task["allowed_nodes"]) == {
            "local", "node001", "node002", "node003",
            "node004", "node005", "node006"}
        signature = task["signature"]
        seed = int(signature.split("/seed-")[1].split("/")[0])
        assert set(task["wait_for_files"]) == {
            str(protocol.calibration_path(seed)),
            *(str(path) for path in protocol.bundle_dependencies_for_seed(seed)),
        }
        assert len(task["wait_for_files"]) == 29


def test_resac_keeps_positive_sign_and_paper_learning_rate():
    command = training_command(0, "resac", None)
    assert command[command.index("--task_num") + 1] == "4"
    assert command[command.index("--test_task_num") + 1] == "4"
    assert command[command.index("--weight_reg") + 1] == "0.01"
    assert command[command.index("--beta_ood") + 1] == "0.01"
    assert command[command.index("--lr") + 1] == "1e-05"


def test_sealed_evaluation_streams_are_disjoint_from_calibration():
    assert set(protocol.EVALUATION_EVENT_SEEDS).isdisjoint(
        protocol.CALIBRATION_EVENT_SEEDS)
    assert protocol.EVALUATION_EVENT_SEEDS == (
        11100, 11200, 11300, 11400, 11500)


def test_analysis_is_not_pinned_to_busy_local_cpu():
    submit = _submit_module()
    rows = submit.candidates("analysis", "high")
    assert len(rows) == 1
    _, task, _ = rows[0]
    assert task["vram"] == 0
    assert set(task["allowed_nodes"]) == {
        "local", "node001", "node002", "node003",
        "node004", "node005", "node006"}
