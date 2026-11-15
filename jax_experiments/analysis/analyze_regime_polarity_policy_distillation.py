"""Aggregate the DAgger policy-distillation screen."""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_audit as audit,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation as trainer,
)


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _event_results(group: str, student_seed: int):
    rows = {}
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        audit.validate_audit(group, student_seed, event_seed)
        result = protocol.read_json(
            protocol.audit_dir(group, student_seed, event_seed)
            / "results.json")
        rows[int(event_seed)] = {
            "switching": {
                row["arm"]: row for row in result["switching"]
            },
            "stationary": {
                (row["arm"], int(row["mode"])): row
                for row in result["stationary"]
            },
        }
    return rows


def _arm_values(events, arm: str):
    return {
        event_seed: [float(value) for value in event["switching"][arm]["returns"]]
        for event_seed, event in events.items()
    }


def _population_values(events, arms):
    output = {}
    for event_seed, event in events.items():
        matrix = np.asarray([
            event["switching"][arm]["returns"] for arm in arms
        ], dtype=np.float64)
        output[event_seed] = list(np.mean(matrix, axis=0))
    return output


def _comparison(candidate, reference) -> dict[str, Any]:
    event_deltas = {}
    paired = []
    candidate_values = []
    reference_values = []
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        left = candidate[int(event_seed)]
        right = reference[int(event_seed)]
        if len(left) != len(right):
            raise ValueError("unpaired policy-distillation audit vectors")
        deltas = [float(a) - float(b) for a, b in zip(left, right)]
        event_deltas[str(event_seed)] = _mean(deltas)
        paired.extend(deltas)
        candidate_values.extend(left)
        reference_values.extend(right)
    return {
        "candidate_mean": _mean(candidate_values),
        "reference_mean": _mean(reference_values),
        "mean_delta": _mean(paired),
        "event_seed_deltas": event_deltas,
        "event_seed_wins": sum(value > 0.0 for value in event_deltas.values()),
        "episode_wins": sum(value > 0.0 for value in paired),
        "n_episodes": len(paired),
    }


def _stationary_summary(events, arm: str) -> dict[str, Any]:
    by_mode = {
        str(mode): _mean(
            value
            for event in events.values()
            for value in event["stationary"][(arm, mode)]["returns"]
        )
        for mode in protocol.MODES
    }
    return {
        "by_mode": by_mode,
        "worst_mode": min(by_mode.values()),
    }


def _student_analysis(group: str, student_seed: int) -> dict[str, Any]:
    events = _event_results(group, student_seed)
    robust_arms = [
        audit.robust_label(source_group, seed)
        for source_group, seed in protocol.controller_keys(group)
    ]
    robust_scores = {
        arm: _mean(
            value
            for values in _arm_values(events, arm).values()
            for value in values
        )
        for arm in robust_arms
    }
    best_robust = max(robust_scores, key=robust_scores.get)
    population = _population_values(events, robust_arms)
    student = _arm_values(events, "student_learned")
    teacher = _arm_values(events, "teacher_learned_median")
    oracle = _arm_values(events, "teacher_oracle_median")
    best = _arm_values(events, best_robust)
    student_vs_population = _comparison(student, population)
    teacher_vs_population = _comparison(teacher, population)
    headroom = float(teacher_vs_population["mean_delta"])
    recovery = (
        float(student_vs_population["mean_delta"]) / headroom
        if headroom > 0.0 else None
    )
    terminated_rate = _mean(
        event["switching"]["student_learned"]["terminated_rate"]
        for event in events.values()
    )
    passed = (
        headroom > 0.0
        and recovery is not None
        and np.isfinite(recovery)
        and recovery >= protocol.MIN_HEADROOM_RECOVERY
        and int(student_vs_population["event_seed_wins"])
        == protocol.MIN_EVENT_SEED_WINS
        and terminated_rate <= protocol.MAX_TERMINATED_RATE
    )
    manifest = trainer.validate_model(group, student_seed)
    return {
        "student_seed": int(student_seed),
        "passed": bool(passed),
        "headroom_recovery": recovery,
        "switching_terminated_rate": terminated_rate,
        "student_vs_individual_robust_population": student_vs_population,
        "teacher_vs_individual_robust_population": teacher_vs_population,
        "student_vs_best_individual_robust": _comparison(student, best),
        "student_vs_teacher": _comparison(student, teacher),
        "oracle_vs_individual_robust_population": _comparison(
            oracle, population),
        "best_individual_robust": {
            "arm": best_robust,
            "mean": robust_scores[best_robust],
        },
        "individual_robust_means": robust_scores,
        "student_stationary": _stationary_summary(
            events, "student_learned"),
        "teacher_stationary": _stationary_summary(
            events, "teacher_learned_median"),
        "final_validation_loss": float(
            manifest["training_phases"][-1]["best_validation_loss"]),
    }


def _group_analysis(group: str) -> dict[str, Any]:
    students = {
        str(seed): _student_analysis(group, seed)
        for seed in protocol.STUDENT_SEEDS
    }
    # This selection is based only on the frozen supervised validation split,
    # never on the strict control audit reported below.
    selected_seed = min(
        protocol.STUDENT_SEEDS,
        key=lambda seed: students[str(seed)]["final_validation_loss"],
    )
    pass_count = sum(row["passed"] for row in students.values())
    selected_pass = bool(students[str(selected_seed)]["passed"])
    return {
        "student_models": students,
        "validation_selected_seed": int(selected_seed),
        "student_seed_passes": int(pass_count),
        "group_pass": bool(
            pass_count >= protocol.MIN_STUDENT_SEED_PASSES
            and selected_pass),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Polarity policy-distillation screen",
        "",
        "This is retrospective algorithm development. Individual robust "
        "controllers, not their invalid action ensemble, are the baseline.",
        "",
    ]
    for group in protocol.TEACHER_GROUPS:
        result = payload["groups"][group]
        lines.extend([
            f"## {group.title()} teacher",
            "",
            "| student seed | student | robust population | delta | "
            "best robust delta | teacher recovery | event wins | pass |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for seed in protocol.STUDENT_SEEDS:
            row = result["student_models"][str(seed)]
            population = row[
                "student_vs_individual_robust_population"]
            best = row["student_vs_best_individual_robust"]
            recovery = row["headroom_recovery"]
            recovery_text = (
                f"{100.0 * recovery:.1f}%"
                if recovery is not None and np.isfinite(recovery)
                else "n/a"
            )
            lines.append(
                f"| {seed} | {population['candidate_mean']:.1f} | "
                f"{population['reference_mean']:.1f} | "
                f"{population['mean_delta']:+.1f} | "
                f"{best['mean_delta']:+.1f} | "
                f"{recovery_text} | "
                f"{population['event_seed_wins']}/3 | "
                f"{'yes' if row['passed'] else 'no'} |"
            )
        lines.extend([
            "",
            f"Validation-selected seed: `{result['validation_selected_seed']}`. "
            f"Student passes: `{result['student_seed_passes']}/3`. "
            f"Group pass: `{'yes' if result['group_pass'] else 'no'}`.",
            "",
        ])
    lines.extend([
        "## Decision",
        "",
        payload["recommendation"],
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    groups = {
        group: _group_analysis(group)
        for group in protocol.TEACHER_GROUPS
    }
    combined = groups["combined"]
    if combined["group_pass"]:
        recommendation = (
            "The combined median teacher can be compressed into one stable "
            "causal student. Freeze the validation-selected combined student "
            "and run a genuinely new five-seed confirmation; do not select a "
            "student from audit return.")
    elif any(groups[group]["group_pass"] for group in ("development", "final")):
        recommendation = (
            "Distillation works only for a controller subset. Diagnose "
            "cross-group teacher incompatibility; do not launch a final "
            "confirmation or select the best audit arm post hoc.")
    else:
        recommendation = (
            "Median ensemble behavior does not survive single-network "
            "compression. Retire offline action distillation and replace the "
            "multi-seed teacher with a jointly trained shared-trunk ensemble.")
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "development_only": True,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "thresholds": {
            "min_event_seed_wins": protocol.MIN_EVENT_SEED_WINS,
            "min_headroom_recovery": protocol.MIN_HEADROOM_RECOVERY,
            "min_student_seed_passes": protocol.MIN_STUDENT_SEED_PASSES,
            "max_terminated_rate": protocol.MAX_TERMINATED_RATE,
        },
        "groups": groups,
        "promotion": bool(combined["group_pass"]),
        "recommendation": recommendation,
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(f"POLICY DISTILLATION ANALYSIS COMPLETE: {protocol.ANALYSIS_ROOT}")
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
