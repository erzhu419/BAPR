"""Audit BAPR construction, deployment, and comparison budgets."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as final,
)
from jax_experiments.analysis import regime_polarity_headroom as estimator_source
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
)


ROOT = final.ROOT
OUTPUT_ROOT = ROOT / "jax_experiments/results_bapr_deployment_budget_v1"
OUTPUT_JSON = OUTPUT_ROOT / "budget.json"
OUTPUT_MD = OUTPUT_ROOT / "analysis.md"
REPORT = ROOT / "reports/bapr_deployment_compression_budget_2026-08-09.md"
CORRECTED_ANALYSIS = (
    ROOT / "jax_experiments/results_regime_polarity_"
    "corrected_baseline_analysis_v2/analysis.json")


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parameter_count(records) -> int:
    return int(sum(math.prod(record["shape"]) for record in records))


def _tree_count(value) -> tuple[int, int]:
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(value)]
    return (
        int(sum(leaf.size for leaf in leaves)),
        int(sum(leaf.nbytes for leaf in leaves)),
    )


def _source_controller_budget() -> dict[str, Any]:
    directories = final.source_bundle_dirs("mode_heads")
    if len(directories) != 20:
        raise ValueError(f"expected 20 teacher bundles, found {len(directories)}")
    total_bytes = 0
    total_steps = 0
    update_count = 0
    for directory in directories:
        manifest = _read(directory / "bundle_manifest.json")
        checkpoint = manifest.get("checkpoint") or {}
        if (
            int(checkpoint.get("total_steps", -1)) != final.FINAL_TOTAL_STEPS
            or int(checkpoint.get("update_count", -1))
            != final.FINAL_UPDATE_COUNT
        ):
            raise ValueError(f"teacher bundle has wrong budget: {directory}")
        total_steps += int(checkpoint["total_steps"])
        update_count += int(checkpoint["update_count"])
        total_bytes += sum(
            path.stat().st_size for path in directory.rglob("*")
            if path.is_file()
        )
    return {
        "controller_count": len(directories),
        "environment_steps": total_steps,
        "gradient_updates": update_count,
        "full_bundle_bytes": total_bytes,
    }


def _estimator_budget() -> dict[str, Any]:
    manifest = _read(final.ensemble.final.MODEL_MANIFEST)
    source_keys = manifest.get("source_bundles") or {}
    if len(source_keys) != 6:
        raise ValueError(f"expected six estimator source bundles: {source_keys}")
    for key in source_keys:
        split, role, seed_label = key.split("/")
        del split
        seed = int(seed_label.removeprefix("seed_"))
        directory = estimator_source.bundle_dir(final.ENV, role, seed)
        source_manifest = _read(directory / "bundle_manifest.json")
        checkpoint = source_manifest.get("checkpoint") or {}
        if (
            int(checkpoint.get("total_steps", -1))
            != estimator_source.FINAL_TOTAL_STEPS
            or int(checkpoint.get("update_count", -1))
            != estimator_source.FINAL_UPDATE_COUNT
        ):
            raise ValueError(f"estimator source has wrong budget: {directory}")
    config = manifest["training_config"]
    train_controllers = 4
    validation_controllers = 2
    train_events = len(manifest["train_event_seeds"])
    validation_events = len(manifest["validation_event_seeds"])
    stationary_per_controller_event = (
        len(final.MODES) * int(config["stationary_steps"]))
    switching_per_controller_event = (
        int(config["switching_episodes"]) * final.MAX_EPISODE_STEPS)
    train_rows = (
        train_controllers * train_events
        * (stationary_per_controller_event + switching_per_controller_event)
    )
    validation_rows = (
        validation_controllers * validation_events
        * (stationary_per_controller_event + switching_per_controller_event)
    )
    return {
        "source_controller_count": len(source_keys),
        "source_environment_steps": (
            len(source_keys) * estimator_source.FINAL_TOTAL_STEPS),
        "source_gradient_updates": (
            len(source_keys) * estimator_source.FINAL_UPDATE_COUNT),
        "data_environment_steps": train_rows + validation_rows,
        "training_rows": train_rows,
        "validation_rows": validation_rows,
        "estimator_updates": int(config["updates"]),
        "parameter_count": _parameter_count(manifest["parameter_leaves"]),
        "parameter_file_bytes": int(manifest["parameter_file"]["size"]),
    }


def _student_budget() -> dict[str, Any]:
    rows = []
    control_per_phase = (
        len(final.CONTROL_VALIDATION_EVENT_SEEDS)
        * final.CONTROL_VALIDATION_SWITCHING_EPISODES
        * final.MAX_EPISODE_STEPS
        * 2
    )
    expected_phases = 1 + final.DAGGER_ROUNDS
    for seed in final.STUDENT_SEEDS:
        path = final.model_manifest("mode_heads", seed)
        manifest = _read(path)
        dataset = manifest["dataset"]
        phases = manifest["training_phases"]
        if len(phases) != expected_phases:
            raise ValueError(f"student {seed} has incomplete phases")
        rows.append({
            "seed": int(seed),
            "training_rows": int(dataset["final_rows"]),
            "validation_rows": int(dataset["validation_rows"]),
            "control_validation_steps": control_per_phase * len(phases),
            "environment_steps": (
                int(dataset["final_rows"])
                + int(dataset["validation_rows"])
                + control_per_phase * len(phases)
            ),
            "gradient_updates": (
                int(final.MODEL_CONFIG["initial_updates"])
                + final.DAGGER_ROUNDS
                * int(final.MODEL_CONFIG["dagger_updates_per_round"])
            ),
            "parameter_count": _parameter_count(
                manifest["parameter_leaves"]),
            "parameter_file_bytes": int(
                manifest["parameter_file"]["size"]),
        })
    invariant = {
        key for key in (
            "training_rows", "validation_rows", "control_validation_steps",
            "environment_steps", "gradient_updates", "parameter_count",
        )
        if len({row[key] for row in rows}) != 1
    }
    if invariant:
        raise ValueError(f"student budgets differ unexpectedly: {invariant}")
    return {
        "per_student": rows[0] | {"seed": "any frozen final seed"},
        "students": rows,
        "aggregate_environment_steps": sum(
            row["environment_steps"] for row in rows),
        "aggregate_gradient_updates": sum(
            row["gradient_updates"] for row in rows),
    }


def _deployment_size(students, estimator) -> dict[str, Any]:
    _patch_flax_variablestate_unpickle()
    robust_path = (
        final.development.ensemble.final.bundle_dir(
            final.ENV, "robust", final.ROBUST_SEED)
        / "checkpoints/params.pkl")
    with robust_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    actor_parameters, actor_bytes = _tree_count(checkpoint["policy"])
    critic_parameters, critic_bytes = _tree_count(checkpoint["critic"])
    student_parameters = int(students["per_student"]["parameter_count"])
    student_file_bytes = int(np.mean([
        row["parameter_file_bytes"] for row in students["students"]
    ]))
    estimator_parameters = int(estimator["parameter_count"])
    estimator_file_bytes = int(estimator["parameter_file_bytes"])
    deployed_parameters = (
        actor_parameters + student_parameters + estimator_parameters)
    deployed_raw_bytes = 4 * deployed_parameters
    deployed_artifact_bytes = (
        actor_bytes + student_file_bytes + estimator_file_bytes)
    teacher_actor_parameters = 20 * actor_parameters
    teacher_actor_bytes = 20 * actor_bytes
    return {
        "robust_actor": {
            "parameter_count": actor_parameters,
            "raw_parameter_bytes": actor_bytes,
        },
        "robust_critic_not_deployed": {
            "parameter_count": critic_parameters,
            "raw_parameter_bytes": critic_bytes,
        },
        "student": {
            "parameter_count": student_parameters,
            "mean_npz_bytes": student_file_bytes,
        },
        "estimator": {
            "parameter_count": estimator_parameters,
            "npz_bytes": estimator_file_bytes,
        },
        "bapr_deployed_stack": {
            "parameter_count": deployed_parameters,
            "raw_float32_bytes": deployed_raw_bytes,
            "mixed_artifact_bytes": deployed_artifact_bytes,
            "size_vs_single_actor": deployed_raw_bytes / actor_bytes,
        },
        "twenty_actor_teacher_bank": {
            "parameter_count": teacher_actor_parameters,
            "raw_float32_bytes": teacher_actor_bytes,
            "compression_ratio_to_bapr_stack": (
                teacher_actor_bytes / deployed_raw_bytes),
        },
    }


def _performance() -> dict[str, Any]:
    analysis = _read(CORRECTED_ANALYSIS)
    methods = analysis["methods"]
    comparisons = analysis["comparisons"]
    return {
        "switching_mean_std": {
            name: [
                float(row["switching_mean"]),
                float(row["switching_std_over_seeds"]),
            ]
            for name, row in methods.items()
        },
        "relative_deltas": {
            name: float(comparisons[name]["relative_delta"])
            for name in (
                "bapr_minus_escp_recurrent",
                "bapr_minus_resac_b0",
                "bapr_minus_sac",
                "bapr_minus_strongest_corrected_baseline",
            )
        },
        "strongest_comparator_interval": comparisons[
            "bapr_minus_strongest_corrected_baseline"
        ]["conservative_two_sample_95pct_interval"],
        "registered_seed_slot_wins_vs_strongest": comparisons[
            "bapr_minus_strongest_corrected_baseline"
        ]["registered_seed_slot_wins"],
        "primary_pass": bool(analysis["primary_pass"]),
        "claim_scope": analysis["claim_scope"],
    }


def _markdown(payload: dict[str, Any]) -> str:
    teacher = payload["construction_budget"]["teacher_bank"]
    estimator = payload["construction_budget"]["estimator"]
    students = payload["construction_budget"]["students"]
    totals = payload["construction_budget"]["totals"]
    deployment = payload["deployment"]
    performance = payload["performance"]
    scores = performance["switching_mean_std"]
    lines = [
        "# BAPR deployment/compression budget audit",
        "",
        "## Frozen empirical result",
        "",
        "| method | switching mean | std over 5 seeds |",
        "|---|---:|---:|",
    ]
    for method in ("bapr", "escp_recurrent", "resac_b0", "sac"):
        mean, std = scores[method]
        lines.append(f"| {method} | {mean:.1f} | {std:.1f} |")
    lines += [
        "",
        f"BAPR improves over recurrent ESCP by "
        f"{100 * performance['relative_deltas']['bapr_minus_escp_recurrent']:.1f}% "
        "and over RE-SAC B0 by "
        f"{100 * performance['relative_deltas']['bapr_minus_resac_b0']:.1f}%. "
        "Against the registered per-seed strongest corrected comparator, the "
        f"gain is only {100 * performance['relative_deltas']['bapr_minus_strongest_corrected_baseline']:.1f}% "
        f"with {performance['registered_seed_slot_wins_vs_strongest']}/5 wins "
        f"and interval {performance['strongest_comparator_interval']}. "
        f"Primary superiority gate: **{performance['primary_pass']}**.",
        "",
        "## Construction budget",
        "",
        "| component | environment interactions | gradient updates |",
        "|---|---:|---:|",
        f"| 20-controller teacher bank | {teacher['environment_steps']:,} | "
        f"{teacher['gradient_updates']:,} |",
        f"| 6 estimator-source controllers | "
        f"{estimator['source_environment_steps']:,} | "
        f"{estimator['source_gradient_updates']:,} |",
        f"| estimator data and fitting | {estimator['data_environment_steps']:,} | "
        f"{estimator['estimator_updates']:,} |",
        f"| 5 final students | {students['aggregate_environment_steps']:,} | "
        f"{students['aggregate_gradient_updates']:,} |",
        f"| BAPR construction total | {totals['bapr_environment_steps']:,} | "
        f"{totals['bapr_gradient_updates']:,} |",
        f"| final single-controller baselines | "
        f"{totals['comparison_baseline_environment_steps']:,} | "
        f"{totals['comparison_baseline_gradient_updates']:,} |",
        f"| construction plus final baselines | "
        f"{totals['experiment_environment_steps']:,} | "
        f"{totals['experiment_gradient_updates']:,} |",
        "",
        "One additional student initialization uses "
        f"{students['per_student']['environment_steps']:,} interactions and "
        f"{students['per_student']['gradient_updates']:,} supervised updates "
        "after the teacher bank and estimator exist. This is a deployment "
        "replication cost, not an end-to-end sample-efficiency number.",
        "",
        "## Deployment footprint",
        "",
        f"- Robust fallback actor: "
        f"{deployment['robust_actor']['parameter_count']:,} parameters.",
        f"- Causal estimator: "
        f"{deployment['estimator']['parameter_count']:,} parameters.",
        f"- Mode-head student: "
        f"{deployment['student']['parameter_count']:,} parameters.",
        f"- Complete deployed BAPR stack: "
        f"{deployment['bapr_deployed_stack']['parameter_count']:,} parameters, "
        f"{deployment['bapr_deployed_stack']['raw_float32_bytes'] / 2**20:.2f} MiB raw float32.",
        f"- Twenty actor-only teachers: "
        f"{deployment['twenty_actor_teacher_bank']['raw_float32_bytes'] / 2**20:.2f} MiB; "
        f"actor-only compression is "
        f"{deployment['twenty_actor_teacher_bank']['compression_ratio_to_bapr_stack']:.2f}x.",
        f"- Versus one SAC actor, BAPR deploys "
        f"{deployment['bapr_deployed_stack']['size_vs_single_actor']:.2f}x "
        "as many float32 parameters. Full training checkpoints are not a fair "
        "deployment-storage comparator.",
        "",
        "## Claim boundary",
        "",
        "The supported framing is deployment-time compression and causal "
        "adaptation on the frozen HalfCheetah actuator-polarity benchmark. "
        "The present evidence does not support end-to-end sample efficiency, "
        "universal MuJoCo superiority, or superiority over a per-seed oracle "
        "choice of corrected baselines. Mechanism v2 and the untouched "
        "persistent-damping headroom screen remain required before expanding "
        "the claim.",
        "",
    ]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    teacher = _source_controller_budget()
    estimator = _estimator_budget()
    students = _student_budget()
    deployment = _deployment_size(students, estimator)
    baseline_steps = 3 * len(final.TRAINING_SEEDS) * final.FINAL_TOTAL_STEPS
    baseline_updates = 3 * len(final.TRAINING_SEEDS) * final.FINAL_UPDATE_COUNT
    bapr_steps = (
        teacher["environment_steps"]
        + estimator["source_environment_steps"]
        + estimator["data_environment_steps"]
        + students["aggregate_environment_steps"]
    )
    bapr_updates = (
        teacher["gradient_updates"]
        + estimator["source_gradient_updates"]
        + estimator["estimator_updates"]
        + students["aggregate_gradient_updates"]
    )
    payload = {
        "schema": "bapr.deployment-compression-budget.v1",
        "status": "complete",
        "construction_budget": {
            "teacher_bank": teacher,
            "estimator": estimator,
            "students": students,
            "totals": {
                "bapr_environment_steps": bapr_steps,
                "bapr_gradient_updates": bapr_updates,
                "comparison_baseline_environment_steps": baseline_steps,
                "comparison_baseline_gradient_updates": baseline_updates,
                "experiment_environment_steps": bapr_steps + baseline_steps,
                "experiment_gradient_updates": bapr_updates + baseline_updates,
            },
        },
        "deployment": deployment,
        "performance": _performance(),
        "claim": (
            "deployment-time causal policy compression; no end-to-end "
            "sample-efficiency or universal-superiority claim"
        ),
    }
    markdown = _markdown(payload)
    final.write_json_atomic(OUTPUT_JSON, payload)
    final.write_text_atomic(OUTPUT_MD, markdown)
    final.write_text_atomic(REPORT, markdown)
    print(
        "BAPR DEPLOYMENT BUDGET COMPLETE: "
        f"steps={bapr_steps} params="
        f"{deployment['bapr_deployed_stack']['parameter_count']}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()
