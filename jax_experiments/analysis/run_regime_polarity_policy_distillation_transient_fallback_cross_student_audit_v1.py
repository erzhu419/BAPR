"""Run one frozen student/event causal-fallback audit."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_cross_student_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_fallback_screen_v1 as evaluator,
)


def validate_audit(student_seed: int, event_seed: int) -> dict:
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    protocol.validate_upstream(student_seed)
    destination = protocol.audit_dir(student_seed, event_seed)
    identity = protocol.identity(student_seed, event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    records = protocol.student_records(student_seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("source_bundles")
        != model_lib.source_records(protocol.TEACHER_GROUP)
        or manifest.get("student_manifest") != records["manifest"]
        or manifest.get("student_parameters") != records["parameters"]
        or manifest.get("frozen_estimator_manifest")
        != protocol.FROZEN_ESTIMATOR_MANIFEST_RECORD
        or manifest.get("frozen_estimator_parameters")
        != protocol.FROZEN_ESTIMATOR_PARAMETER_RECORD
        or manifest.get("result_file")
        != protocol.file_record(destination / "results.json")
    ):
        raise ValueError(f"invalid cross-student fallback audit {destination}")
    result = protocol.read_json(destination / "results.json")
    evaluator._validate_result(
        result, identity, protocol.ARMS, protocol.EVENT_SCHEMA)
    return manifest


def evaluate_event(student_seed: int, event_seed: int) -> list[dict]:
    protocol.validate_upstream(student_seed)
    teacher = model_lib.load_teacher(protocol.TEACHER_GROUP)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    student, student_params, _ = model_lib.load_student(
        protocol.TEACHER_GROUP,
        student_seed,
        teacher.obs_dim,
        teacher.act_dim,
    )
    student_action = model_lib.build_student_action(student)
    rows = []
    for arm in protocol.ARMS:
        rows.append(evaluator.evaluate_switching_arm(
            teacher.config,
            teacher,
            estimator,
            student_action,
            student_params,
            arm,
            event_seed,
        ))
        print(
            f"cross-student fallback student={student_seed} "
            f"event={event_seed} arm={arm} complete",
            flush=True,
        )
    return rows


def run(student_seed: int, event_seed: int) -> None:
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(student_seed, event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(student_seed, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"CROSS-STUDENT FALLBACK ALREADY COMPLETE: {destination}")
            return

    identity = protocol.identity(student_seed, event_seed)
    result = {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "switching": evaluate_event(student_seed, event_seed),
    }
    if destination.exists() or destination.is_symlink():
        shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(temporary_path / "results.json", result)
        records = protocol.student_records(student_seed)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": identity,
            "source_bundles": model_lib.source_records(protocol.TEACHER_GROUP),
            "student_manifest": records["manifest"],
            "student_parameters": records["parameters"],
            "frozen_estimator_manifest":
                protocol.FROZEN_ESTIMATOR_MANIFEST_RECORD,
            "frozen_estimator_parameters":
                protocol.FROZEN_ESTIMATOR_PARAMETER_RECORD,
            "frozen_selection_manifest":
                protocol.FROZEN_SELECTION_MANIFEST_RECORD,
            "frozen_transient_analysis":
                protocol.FROZEN_TRANSIENT_ANALYSIS_RECORD,
            "result_file": protocol.file_record(temporary_path / "results.json"),
        }
        protocol.write_json_atomic(
            temporary_path / "audit_manifest.json", manifest)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
    validate_audit(student_seed, event_seed)
    print(f"CROSS-STUDENT FALLBACK COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.student_seed, args.event_seed)


if __name__ == "__main__":
    main()
