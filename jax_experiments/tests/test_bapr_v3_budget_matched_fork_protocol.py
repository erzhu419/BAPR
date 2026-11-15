import importlib.util
import hashlib
import json
import pickle
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as protocol
from jax_experiments.analysis import recover_bapr_v3_budget_matched_fork as recovery
from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as fork_audit,
)


def _boundary(rollout_sha, observation_sha, task_sha, *, warmstarted):
    return {
        "conditioned_warmstarted": warmstarted,
        "physical_rollout": {
            "sha256": rollout_sha,
            "field_sha256": {
                "observation": observation_sha,
                "task_id": task_sha,
            },
        },
        "policy_equivalence": {
            "pass": True,
            "finite": True,
            "max_abs_mean_diff": 0.0,
            "max_abs_log_std_diff": 0.0,
        },
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _audit_overlay_fixture(tmp_path, monkeypatch, *, changed=()):
    monkeypatch.setattr(fork_audit, "ROOT", tmp_path)
    files = {}
    for relative in fork_audit.AUDIT_ORCHESTRATOR_MODULES:
        producer_payload = f"producer:{relative}".encode()
        live_payload = (
            f"live:{relative}".encode()
            if relative in changed else producer_payload)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(live_payload)
        files[relative] = {
            "sha256": _sha256(producer_payload),
            "size": len(producer_payload),
        }
    if fork_audit.AUDIT_VALIDATOR_MODULE in changed:
        live_validator_hash = _sha256(
            (tmp_path / fork_audit.AUDIT_VALIDATOR_MODULE).read_bytes())
        monkeypatch.setattr(
            fork_audit,
            "CATEGORICAL_AUDIT_VALIDATOR_SHA256",
            frozenset({live_validator_hash}),
        )
        monkeypatch.setattr(
            fork_audit,
            "CATEGORICAL_AUDIT_VALIDATOR_LEGACY_SHA256",
            frozenset(),
        )
    validator_relative = (
        "jax_experiments/analysis/run_bapr_v3_budget_matched_fork.py")
    validator_path = tmp_path / validator_relative
    validator_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path.write_bytes(b"categorical boundary validator")
    source = {
        "snapshot_sha256": "a" * 64,
        "sha256": "a" * 64,
        "archive": {
            "path": "provenance/source_snapshot.tar.gz",
            "sha256": "b" * 64,
            "size": 1,
        },
        "files": files,
    }
    manifest = {
        "identity": {
            "family": "deterministic_mean",
            "env": "Ant-v2",
            "seed": 0,
            "policy_variant": "cat_lcb",
        },
        "source": source,
        "finalization": {
            "mode": "posthoc-categorical-boundary-validator-v1",
            "training_reentered": False,
            "original_training_source_sha256": "a" * 64,
            "original_source_archive_sha256": "b" * 64,
            "validator_file": validator_relative,
            "validator_file_sha256": (
                sorted(fork_audit.CATEGORICAL_BOUNDARY_VALIDATOR_SHA256)[0]),
        },
        "resume_boundary": {
            "physical_rollout_validation": (
                "categorical_policy_equivalence"),
            "physical_rollout_accepted": True,
            "task_schedule_equal": True,
            "policy_canary_exact": True,
        },
    }
    return source, manifest


def test_audit_orchestrator_preserves_exact_producer_manifest_shape(
    tmp_path, monkeypatch,
):
    source, _manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch, changed=())

    record = fork_audit.expected_orchestrator_source_provenance(source)

    assert set(record) == set(fork_audit.AUDIT_ORCHESTRATOR_MODULES)
    assert all(len(value) == 64 for value in record.values())


def test_audit_orchestrator_accepts_only_categorical_validator_overlay(
    tmp_path, monkeypatch,
):
    source, manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch,
        changed=(fork_audit.AUDIT_VALIDATOR_MODULE,))

    record = fork_audit.expected_orchestrator_source_provenance(
        source, manifest)

    assert record["mode"] == (
        fork_audit.CATEGORICAL_VALIDATOR_OVERLAY_MODE)
    assert record["changed_files"] == [fork_audit.AUDIT_VALIDATOR_MODULE]
    assert record["authorization"]["policy_variant"] == "cat_lcb"
    assert record["authorization"]["training_reentered"] is False


def test_audit_orchestrator_rejects_extra_changed_module(tmp_path, monkeypatch):
    source, manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch,
        changed=(
            fork_audit.AUDIT_VALIDATOR_MODULE,
            fork_audit.AUDIT_ORCHESTRATOR_MODULES[0],
        ))

    with pytest.raises(ValueError, match="changed modules"):
        fork_audit.expected_orchestrator_source_provenance(source, manifest)


def test_audit_orchestrator_rejects_training_reentry(tmp_path, monkeypatch):
    source, manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch,
        changed=(fork_audit.AUDIT_VALIDATOR_MODULE,))
    manifest["finalization"]["training_reentered"] = True

    with pytest.raises(ValueError, match="training-reentered"):
        fork_audit.expected_orchestrator_source_provenance(source, manifest)


def test_audit_orchestrator_rejects_unknown_finalizer_hash(
    tmp_path, monkeypatch,
):
    source, manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch,
        changed=(fork_audit.AUDIT_VALIDATOR_MODULE,))
    manifest["finalization"]["validator_file_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="unrecognized categorical"):
        fork_audit.expected_orchestrator_source_provenance(source, manifest)


def test_live_audit_overlay_requires_authoritative_manifest_env(
    tmp_path, monkeypatch,
):
    source, manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch,
        changed=(fork_audit.AUDIT_VALIDATOR_MODULE,))
    manifest_path = tmp_path / "pair" / "provenance" / "pair_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv(
        fork_audit.AUDIT_PAIR_MANIFEST_ENV, str(manifest_path.resolve()))

    record = fork_audit.validate_live_audit_modules(source)

    assert record["mode"] == (
        fork_audit.CATEGORICAL_VALIDATOR_OVERLAY_MODE)


def test_recorded_audit_overlay_accepts_only_known_generator_hash(
    tmp_path, monkeypatch,
):
    source, manifest = _audit_overlay_fixture(
        tmp_path, monkeypatch,
        changed=(fork_audit.AUDIT_VALIDATOR_MODULE,))
    record = fork_audit.expected_orchestrator_source_provenance(
        source, manifest)

    fork_audit.validate_recorded_orchestrator_source_provenance(
        record, source, manifest)

    record["live"][fork_audit.AUDIT_VALIDATOR_MODULE] = "d" * 64
    with pytest.raises(ValueError, match="unrecognized recorded"):
        fork_audit.validate_recorded_orchestrator_source_provenance(
            record, source, manifest)


def test_categorical_boundary_accepts_exact_canary_and_task_schedule():
    robust = _boundary("a" * 64, "b" * 64, "c" * 64, warmstarted=False)
    oracle = _boundary("d" * 64, "e" * 64, "c" * 64, warmstarted=True)

    proof = protocol.boundary_rollout_proof(
        {"policy_variant": "cat_lcb"}, robust, oracle)

    assert proof["physical_rollout_validation"] == (
        "categorical_policy_equivalence")
    assert proof["physical_rollout_accepted"] is True
    assert proof["physical_rollout_equal"] is False
    assert proof["field_sha256_equal"] is False
    assert proof["task_schedule_equal"] is True
    assert proof["policy_canary_exact"] is True


def test_direct_boundary_rollout_mismatch_still_fails_closed():
    robust = _boundary("a" * 64, "b" * 64, "c" * 64, warmstarted=False)
    oracle = _boundary("d" * 64, "e" * 64, "c" * 64, warmstarted=True)

    with pytest.raises(RuntimeError, match="PROTOCOL_INTEGRITY"):
        protocol.boundary_rollout_proof({}, robust, oracle)


def test_categorical_boundary_task_schedule_mismatch_fails_closed():
    robust = _boundary("a" * 64, "b" * 64, "c" * 64, warmstarted=False)
    oracle = _boundary("d" * 64, "e" * 64, "f" * 64, warmstarted=True)

    with pytest.raises(RuntimeError, match="mode schedules"):
        protocol.boundary_rollout_proof(
            {"policy_variant": "cat_lcb"}, robust, oracle)


def test_tampered_pinned_shared_base_fails_closed(tmp_path):
    pair = tmp_path / "pair"
    (pair / "provenance").mkdir(parents=True)
    checkpoint = {
        "iteration": protocol.BASE_FINAL_ITERATION,
        "next_iteration": protocol.BASE_NEXT_ITERATION,
        "total_steps": protocol.BASE_TOTAL_STEPS,
        "update_count": protocol.BASE_UPDATE_COUNT,
        "update_count_error": "",
        "replay_size": 1_000_000,
        "replay_ptr": 800_000,
        "logger_key_count": 1,
    }
    state = {
        "runtime": {},
        "source": {},
        "shared_base_snapshot_sha256": "a" * 64,
        "shared_base_manifest_sha256": "c" * 64,
    }
    snapshot = {"sha256": "b" * 64, "files": {}}
    signature = {
        "checkpoint_loaded": False,
        "start_iteration": 0,
        "total_steps_at_start": 0,
    }
    with (
        mock.patch.object(protocol, "branch_checkpoint_iteration", return_value=699),
        mock.patch.object(protocol, "validate_checkpoint", return_value=checkpoint),
        mock.patch.object(
            protocol, "validate_base_protocol_signature", return_value=signature),
        mock.patch.object(protocol, "run_snapshot_manifest", return_value=snapshot),
        mock.patch.object(protocol, "atomic_write_json") as write_manifest,
        mock.patch.object(protocol, "update_state") as update_state,
    ):
        with pytest.raises(RuntimeError, match="immutable shared-base snapshot"):
            protocol.ensure_base(
                pair, "deterministic_mean", "Ant-v2", 0, state)
    write_manifest.assert_not_called()
    update_state.assert_not_called()


def test_categorical_variant_is_provenanced_and_applied_to_all_forks(tmp_path):
    values = protocol.common_training_values(
        "mean_variance", "Ant-v2", 0, tmp_path, "shared_base",
        "cat_anchor_0p1")
    command = " ".join(values)
    assert "--bapr_v2_policy_mode categorical_expert" in command
    assert "--bapr_v2_num_experts 4" in command
    assert "--bapr_v2_action_deviation_weight 0.1" in command
    assert "--bapr_v2_actor_objective mean" in command
    assert protocol.pair_identity(
        "mean_variance", "Ant-v2", 0, "cat_anchor_0p1") == {
            "family": "mean_variance",
            "env": "Ant-v2",
            "seed": 0,
            "policy_variant": "cat_anchor_0p1",
        }
    assert protocol.pair_identity(
        "mean_variance", "Ant-v2", 0, "direct") == {
            "family": "mean_variance", "env": "Ant-v2", "seed": 0}


def test_valid_complete_reentry_is_read_only(tmp_path):
    pair = tmp_path / "pair"
    manifest_path = pair / protocol.PAIR_MANIFEST_REL
    manifest_path.parent.mkdir(parents=True)
    sentinel_path = pair / protocol.COMPLETE_SENTINEL_NAME
    manifest_path.write_bytes(b"manifest")
    sentinel_path.write_bytes(b"sentinel")
    identity = {"family": "deterministic_mean", "env": "Ant-v2", "seed": 0}
    runtime = {"sha256": "r"}
    source = {"sha256": "s", "files": {}}
    manifest = {"identity": identity, "runtime": runtime, "source": source}
    state = {
        "phase": "complete",
        "identity": identity,
        "runtime": runtime,
        "source": source,
        "pair_manifest_sha256": "f" * 64,
    }
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (manifest_path, sentinel_path)
    }
    with (
        mock.patch.object(
            protocol, "validate_complete_artifacts",
            return_value={
                "manifest": manifest, "sentinel": {}, "sha256": "f" * 64}),
        mock.patch.object(protocol, "update_state") as update_state,
    ):
        assert protocol.completed_pair_on_reentry(pair, state) is manifest
    update_state.assert_not_called()
    assert before == {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (manifest_path, sentinel_path)
    }


def test_corrupt_complete_sentinel_fails_before_deep_validation(tmp_path):
    pair = tmp_path / protocol.pair_name(
        "deterministic_mean", "Ant-v2", 0)
    manifest_path = pair / protocol.PAIR_MANIFEST_REL
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "schema": protocol.SCHEMA,
        "status": "complete",
        "completed_at": "2026-07-16T00:00:00+00:00",
        "shared_base": {"snapshot_sha256": "a" * 64},
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    sentinel = {
        "schema": "bapr.pair-checkpoint-complete.v1",
        "status": "complete",
        "pair_manifest": str(protocol.PAIR_MANIFEST_REL),
        "pair_manifest_sha256": protocol.sha256_file(manifest_path),
        "shared_base_snapshot_sha256": "a" * 64,
        "final_next_iteration": protocol.FINAL_NEXT_ITERATION,
        "final_total_steps": protocol.FINAL_TOTAL_STEPS - 1,
        "completed_at": manifest["completed_at"],
    }
    with (pair / protocol.COMPLETE_SENTINEL_NAME).open("wb") as handle:
        pickle.dump(sentinel, handle)
    with pytest.raises(RuntimeError, match="wrong final budget"):
        protocol.validate_complete_artifacts(pair)


def test_submit_spec_never_ignores_initial_resume_scan_errors():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_budget_matched_fork.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_budget_matched_fork", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    task = module.task_spec(
        "deterministic_mean", "Ant-v2", SimpleNamespace(priority="high"))
    assert task["allow_initial_resume_scan_error"] is False
    assert task["reroute_on_node_down"] is False
    assert "-m jax_experiments.analysis.run_bapr_v3_budget_matched_fork" in task["cmd"]


def test_ant_specialization_submitter_keeps_large_pair_remote():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_ant_specialization.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_ant_specialization", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))

    task = module.task_spec("cat_mean", "mean_variance", "high")
    assert task["allow_initial_resume_scan_error"] is False
    assert task["reroute_on_node_down"] is False
    assert "result_dir" not in task
    assert "local_result_dir" not in task
    assert "--policy-variant cat_mean" in task["cmd"]
    assert task["ckpt_dir"].endswith(
        "cat_mean/budget_fork_v2_mean_variance_Ant_s0")


def test_ant_specialization_finalize_is_pinned_and_cannot_train():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_ant_specialization_finalize.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_ant_specialization_finalize", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))

    task = module.task_spec(SimpleNamespace(
        variant="cat_lcb",
        family="deterministic_mean",
        node="local",
        gpu_idx=0,
    ))

    assert task["require_node"] == "local"
    assert task["require_gpu_idx"] == 0
    assert task["vram"] == 512
    assert task["reroute_on_node_down"] is False
    assert task["allow_initial_resume_scan_error"] is False
    assert "VALIDATOR SYNC OK" in task["cmd"]
    assert "run_bapr_v3_budget_matched_fork.py" in task["cmd"]
    assert "analyze_bapr_v3_budget_matched_fork_audit.py" in task["cmd"]
    assert "--resume --finalize-existing" in task["cmd"]
    assert "result_dir" not in task
    assert "local_result_dir" not in task


def test_ant_specialization_audit_is_pinned_and_uses_archived_evaluator():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_ant_specialization_audit.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_ant_specialization_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))

    snapshot = root / module.AUDIT_VALIDATOR_SNAPSHOT_RELATIVE
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == (
        module.AUDIT_VALIDATOR_SNAPSHOT_SHA256)

    task = module.task_spec(
        "cat_anchor_0p1", "mean_variance", 1002, "node007", "high")

    assert task["require_node"] == "node007"
    assert task["vram"] == 6000
    assert task["reroute_on_node_down"] is False
    assert task["allow_initial_resume_scan_error"] is False
    assert task["ckpt_glob"] == protocol.COMPLETE_SENTINEL_NAME
    assert task["ckpt_dir"].endswith(
        "cat_anchor_0p1/budget_fork_v2_mean_variance_Ant_s0")
    assert task["result_dir"].endswith(
        "cat_anchor_0p1/mean_variance/Ant/event_seed_1002")
    assert "AUDIT VALIDATOR SYNC OK" in task["cmd"]
    assert "audit-validator-sync.{os.getpid()}" in task["cmd"]
    assert "BAPR_AUDIT_PAIR_MANIFEST=" in task["cmd"]
    assert "run_bapr_v3_budget_matched_fork_audit_group" in task["cmd"]
    assert "flock --exclusive" in task["cmd"]
    assert "${CUDA_VISIBLE_DEVICES:-unknown}" in task["cmd"]
    assert "final_task_sweep" not in task["cmd"]

    non_node007_task = module.task_spec(
        "cat_anchor_0p1", "mean_variance", 1002, "jtl311linux", "high")
    assert non_node007_task["vram"] == 4200
    assert "flock --exclusive" not in non_node007_task["cmd"]

    local_task = module.task_spec(
        "cat_lcb", "deterministic_mean", 1002, "local", "high")
    assert "AUDIT VALIDATOR SYNC OK" not in local_task["cmd"]

    ready, reason, node = module.resolve_pair_node(
        [{
            "id": "t1",
            "signature": module.finalize_signature(
                "cat_anchor_0p1", "mean_variance"),
            "status": "done",
            "node": "node007",
        }],
        "cat_anchor_0p1", "mean_variance",
    )
    assert ready is True
    assert node == "node007"
    assert "t1" in reason


def test_ant_specialization_analysis_is_pinned_cpu_only_and_small_output():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_ant_specialization_analysis.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_ant_specialization_analysis", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))

    task = module.task_spec(
        "cat_anchor_0p1", "mean_variance", "node007", "high")
    assert task["require_node"] == "node007"
    assert task["vram"] == 0
    assert task["reroute_on_node_down"] is False
    assert task["allow_initial_resume_scan_error"] is False
    assert task["ckpt_glob"] == protocol.COMPLETE_SENTINEL_NAME
    assert task["result_dir"].endswith(
        "cat_anchor_0p1/mean_variance/Ant")
    assert "results_bapr_v3_ant_specialization_analysis_v1" in task["result_dir"]
    assert "JAX_PLATFORMS=cpu" in task["cmd"]
    assert "ANT SPECIALIZATION ANALYZER SYNC OK" in task["cmd"]
    assert "analysis-sync.{os.getpid()}" in task["cmd"]
    assert "--json-output" in task["cmd"]
    assert "results_bapr_v3_ant_specialization_v1" in task["ckpt_dir"]
    assert "results_bapr_v3_ant_specialization_v1" not in task["result_dir"]

    finalize = {
        "id": "t-finalize",
        "signature": module.group_submit.finalize_signature(
            "cat_anchor_0p1", "mean_variance"),
        "status": "done",
        "node": "node007",
    }
    groups = [{
        "id": f"t-{event_seed}",
        "signature": module.group_submit.signature(
            "cat_anchor_0p1", "mean_variance", event_seed),
        "status": "done",
    } for event_seed in module.audit.DEFAULT_EVENT_SEEDS]
    ready, reason, node = module.audit_readiness(
        [finalize, *groups], "cat_anchor_0p1", "mean_variance")
    assert ready is True
    assert node == "node007"
    assert "all five" in reason

    groups[-1]["status"] = "running"
    ready, reason, node = module.audit_readiness(
        [finalize, *groups], "cat_anchor_0p1", "mean_variance")
    assert ready is False
    assert node == "node007"
    assert str(module.audit.DEFAULT_EVENT_SEEDS[-1]) in reason


def test_archive_resume_requires_matching_identity_and_complete_base(tmp_path):
    pair = tmp_path / "pair"
    pair.mkdir()
    state = {
        "identity": {
            "family": "mean_variance",
            "env": "HalfCheetah-v2",
            "seed": 0,
        },
    }
    with (pair / "protocol_checkpoint.pkl").open("wb") as handle:
        pickle.dump(state, handle)
    source = ({"sha256": "a" * 64}, pair / "source.tar.gz")

    with (
        mock.patch.object(recovery, "validate_checkpoint") as validate_checkpoint,
        mock.patch.object(
            recovery, "validate_source_archive", return_value=source),
    ):
        result = recovery.validate_archive_resume(
            pair, "mean_variance", "HalfCheetah-v2", 0)

    assert result == source
    validate_checkpoint.assert_called_once_with(
        pair / "shared_base", recovery.BASE_ITERATION, recovery.BASE_STEPS)


def test_archive_resume_rejects_cross_protocol_identity(tmp_path):
    pair = tmp_path / "pair"
    pair.mkdir()
    with (pair / "protocol_checkpoint.pkl").open("wb") as handle:
        pickle.dump(
            {"identity": {
                "family": "deterministic_mean",
                "env": "HalfCheetah-v2",
                "seed": 0,
            }},
            handle,
        )

    with pytest.raises(RuntimeError, match="protocol identity"):
        recovery.validate_archive_resume(
            pair, "mean_variance", "HalfCheetah-v2", 0)


def test_recovery_submit_spec_is_inline_pinned_and_explicit_for_training():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_budget_matched_fork_recovery.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_budget_matched_fork_recovery", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))

    task = module.task_spec(SimpleNamespace(
        family="mean_variance",
        env="HalfCheetah-v2",
        node="jtl110gpu",
        gpu_idx=0,
        resume_partial=True,
    ))

    assert task["require_node"] == "jtl110gpu"
    assert task["require_gpu_idx"] == 0
    assert task["reroute_on_node_down"] is False
    assert task["allow_initial_resume_scan_error"] is False
    assert "--resume-partial" in task["cmd"]
    assert "<bapr-v3-fork-recovery>" in task["cmd"]


def test_audit_submitter_accepts_one_completed_v4_recovery_as_remote_evidence():
    root = protocol.ROOT
    scripts = root / "scripts"
    module_path = scripts / "submit_bapr_v3_budget_matched_fork_audit.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_budget_matched_fork_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))

    tasks = [
        {
            "signature": module.producer_signature(
                "deterministic_mean", "HalfCheetah-v2"),
            "status": "failed",
            "node": "jtl110gpu2",
        },
        {
            "signature": module.recovery_signature(
                "deterministic_mean", "HalfCheetah-v2"),
            "status": "done",
            "node": "jtl110gpu2",
        },
    ]
    with mock.patch.object(
        module, "producer_ready_local", return_value=(False, "not local"),
    ):
        ready, reason, node = module.resolve_producer(
            tasks, "deterministic_mean", "HalfCheetah-v2")

    assert ready is True
    assert node == "jtl110gpu2"
    assert "source-archive recovery" in reason
