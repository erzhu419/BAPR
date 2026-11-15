from __future__ import annotations

import sys
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_fresh_bank_estimator_confirmation_v17 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v17 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v17 as specialist_runner,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_regime_polarity_fresh_bank_estimator_confirmation_v17 as submitter


def test_v17_protocol_splits_and_confirmation_gates_are_frozen():
    protocol.assert_protocol_integrity()
    assert protocol.TRAINING_SEEDS == (
        81003, 81021, 81039, 81057, 81079)
    assert len(set(protocol.CALIBRATION_EVENT_SEEDS)) == 3
    assert len(set(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS)) == 3
    assert len(set(protocol.SWITCHING_EVENT_SEEDS)) == 3
    assert protocol.REQUIRED_SEED_PASSES == 4
    assert protocol.REQUIRED_V16_SEED_WINS == 4
    assert protocol.REQUIRED_MECHANISM_SEED_PASSES == 4
    assert protocol.ARMS == (
        "robust_sac",
        "true_mode_safe_utility",
        "delayed_oracle_4_safe_utility",
        "frozen_v5_posterior_map",
        "switch_weighted_v16_posterior_map",
    )


def test_v17_thin_runners_target_v17_paths_without_changing_budget():
    source_command = source_runner.training_command(81003)
    assert str(protocol.source_run_dir(81003).parent) in source_command
    assert str(protocol.SOURCE_NEXT_ITERATION) in source_command
    specialist_command = specialist_runner.training_command(81003, 0)
    assert str(protocol.run_dir("actor_only", 81003, 0).parent) in specialist_command
    assert str(protocol.FINAL_NEXT_ITERATION) in specialist_command
    assert str(protocol.SOURCE_NEXT_ITERATION) in specialist_command


def test_v17_specialist_validator_accepts_the_audit_loader_contract():
    calls = []
    original = specialist_runner.base.validate_bundle
    try:
        specialist_runner.base.validate_bundle = (
            lambda seed, mode: calls.append((seed, mode)) or {"ok": True})
        assert specialist_runner.validate_bundle(81003, 2) == {"ok": True}
        assert specialist_runner.validate_bundle(
            "actor_only", 81003, 2) == {"ok": True}
    finally:
        specialist_runner.base.validate_bundle = original
    assert calls == [(81003, 2), (81003, 2)]


def test_v17_scheduler_graph_has_25_gpu_producers_and_6_cpu_tasks():
    rows = submitter.candidates("high")
    assert len(rows) == 31
    assert len({signature for signature, _, _ in rows}) == 31
    gpu = [spec for _, spec, _ in rows if spec["vram"] > 0]
    cpu = [spec for _, spec, _ in rows if spec["vram"] == 0]
    assert len(gpu) == 25
    assert len(cpu) == 6
    assert all(spec["vram"] == 1200 for spec in gpu)
    assert all(spec["allowed_nodes"] == submitter.GPU_NODES for spec in gpu)
    assert all("ckpt_dir" in spec for spec in gpu)
    assert all(
        Path(spec["result_dir"]).name != "checkpoints" for spec in gpu)
    assert all("ckpt_dir" not in spec for spec in cpu)


def test_v17_scheduler_dependencies_are_seed_local_and_complete():
    rows = {
        signature: spec
        for signature, spec, _ in submitter.candidates("high")
    }
    for seed in protocol.TRAINING_SEEDS:
        source = rows[submitter.source_signature(seed)]
        assert all(
            str(path) not in source["wait_for_files"]
            for path in protocol.source_required_paths(seed)
        )
        for mode in protocol.MODES:
            specialist = rows[submitter.specialist_signature(seed, mode)]
            assert set(map(str, protocol.source_required_paths(seed))) <= set(
                specialist["wait_for_files"])
        audit = rows[submitter.audit_signature(seed)]
        expected = {
            *map(str, protocol.source_required_paths(seed)),
            *(
                str(path)
                for mode in protocol.MODES
                for path in protocol.bundle_required_paths(
                    "actor_only", seed, mode)
            ),
            str(protocol.estimator_parent.MODEL_MANIFEST),
            str(protocol.estimator_parent.MODEL_PATH),
            str(protocol.v5_parent.MODEL_MANIFEST),
            str(protocol.v5_parent.MODEL_PATH),
        }
        assert expected <= set(audit["wait_for_files"])
    aggregate = rows[submitter.analysis_signature()]
    assert {
        str(protocol.audit_manifest(seed))
        for seed in protocol.TRAINING_SEEDS
    } <= set(aggregate["wait_for_files"])


def test_v17_wait_files_are_reachable_from_staging_roots():
    for _, spec, _ in submitter.candidates("high"):
        roots = [Path(path) for path in spec["stage_input_paths"]]
        for value in spec["wait_for_files"]:
            path = Path(value)
            assert any(
                path == root or path.is_relative_to(root) for root in roots
            ), (path, roots)
