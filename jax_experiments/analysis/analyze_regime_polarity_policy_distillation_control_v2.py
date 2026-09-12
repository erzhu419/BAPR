"""Aggregate the closed-loop, return-aware compression development sweep."""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_control_audit as audit,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _event_results(variant: str, student_seed: int):
    rows = {}
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        audit.validate_audit(variant, student_seed, event_seed)
        result = protocol.read_json(
            protocol.audit_dir(variant, student_seed, event_seed)
            / "results.json")
        rows[int(event_seed)] = {
            "switching": {row["arm"]: row for row in result["switching"]},
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
            raise ValueError("unpaired closed-loop compression audit vectors")
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
    return {"by_mode": by_mode, "worst_mode": min(by_mode.values())}


def _student_analysis(variant: str, student_seed: int) -> dict[str, Any]:
    events = _event_results(variant, student_seed)
    robust_arms = [
        audit.robust_label(source_group, seed)
        for source_group, seed in protocol.controller_keys(variant)
    ]
    if protocol.FIXED_ROBUST_ARM not in robust_arms:
        raise ValueError("preregistered fixed robust arm is absent")
    robust_scores = {
        arm: _mean(
            value
            for values in _arm_values(events, arm).values()
            for value in values
        )
        for arm in robust_arms
    }
    population = _population_values(events, robust_arms)
    student = _arm_values(events, "student_learned")
    teacher = _arm_values(events, "teacher_learned_median")
    oracle = _arm_values(events, "teacher_oracle_median")
    fixed = _arm_values(events, protocol.FIXED_ROBUST_ARM)
    student_population = _comparison(student, population)
    teacher_population = _comparison(teacher, population)
    student_fixed = _comparison(student, fixed)
    student_teacher = _comparison(student, teacher)
    headroom = float(teacher_population["mean_delta"])
    recovery = (
        float(student_population["mean_delta"]) / headroom
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
        and int(student_population["event_seed_wins"])
        >= protocol.MIN_EVENT_WINS_VS_POPULATION
        and int(student_fixed["event_seed_wins"])
        >= protocol.MIN_EVENT_WINS_VS_FIXED_ROBUST
        and float(student_teacher["mean_delta"])
        >= -protocol.MAX_MEAN_STUDENT_TEACHER_GAP
        and terminated_rate <= protocol.MAX_TERMINATED_RATE
    )
    manifest = trainer.validate_model(variant, student_seed)
    return {
        "student_seed": int(student_seed),
        "passed": bool(passed),
        "headroom_recovery": recovery,
        "switching_terminated_rate": terminated_rate,
        "student_vs_individual_robust_population": student_population,
        "teacher_vs_individual_robust_population": teacher_population,
        "student_vs_fixed_robust": student_fixed,
        "student_vs_teacher": student_teacher,
        "oracle_vs_fixed_robust": _comparison(oracle, fixed),
        "fixed_robust": {
            "arm": protocol.FIXED_ROBUST_ARM,
            "mean": robust_scores[protocol.FIXED_ROBUST_ARM],
        },
        "individual_robust_means": robust_scores,
        "student_stationary": _stationary_summary(events, "student_learned"),
        "teacher_stationary": _stationary_summary(
            events, "teacher_learned_median"),
        "selected_phase": int(manifest["selected_phase"]),
        "control_validation_score": float(
            manifest["selected_control_validation_score"]),
        "control_validation": manifest["selected_control_validation"],
    }


def _variant_analysis(variant: str) -> dict[str, Any]:
    students = {
        str(seed): _student_analysis(variant, seed)
        for seed in protocol.STUDENT_SEEDS
    }
    # Audit returns are never used for model selection. This seed is chosen
    # solely from each model's independent control-validation split.
    selected_seed = max(
        protocol.STUDENT_SEEDS,
        key=lambda seed: students[str(seed)]["control_validation_score"],
    )
    pass_count = sum(row["passed"] for row in students.values())
    selected_pass = bool(students[str(selected_seed)]["passed"])
    return {
        "student_models": students,
        "control_validation_selected_seed": int(selected_seed),
        "control_validation_selected_score": float(
            students[str(selected_seed)]["control_validation_score"]),
        "student_seed_passes": int(pass_count),
        "selected_student_pass": selected_pass,
        "variant_pass": bool(
            pass_count >= protocol.MIN_STUDENT_SEED_PASSES
            and selected_pass),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Closed-loop policy-compression v2 development result",
        "",
        "This is development-only. The five sealed confirmation events were "
        "not reused. Individual robust controllers, including the fixed "
        "`robust_final_seed_719`, are the valid baselines.",
        "",
    ]
    for variant in protocol.VARIANTS:
        result = payload["variants"][variant]
        lines.extend([
            f"## {variant}",
            "",
            "| student seed | student | robust population | population delta | "
            "fixed-719 delta | teacher gap | recovery | event wins vs 719 | pass |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for seed in protocol.STUDENT_SEEDS:
            row = result["student_models"][str(seed)]
            population = row["student_vs_individual_robust_population"]
            fixed = row["student_vs_fixed_robust"]
            teacher = row["student_vs_teacher"]
            recovery = row["headroom_recovery"]
            recovery_text = (
                f"{100.0 * recovery:.1f}%"
                if recovery is not None and np.isfinite(recovery) else "n/a")
            lines.append(
                f"| {seed} | {population['candidate_mean']:.1f} | "
                f"{population['reference_mean']:.1f} | "
                f"{population['mean_delta']:+.1f} | "
                f"{fixed['mean_delta']:+.1f} | "
                f"{teacher['mean_delta']:+.1f} | {recovery_text} | "
                f"{fixed['event_seed_wins']}/3 | "
                f"{'yes' if row['passed'] else 'no'} |"
            )
        lines.extend([
            "",
            f"Control-validation-selected seed: "
            f"`{result['control_validation_selected_seed']}`. Student passes: "
            f"`{result['student_seed_passes']}/3`. Variant pass: "
            f"`{'yes' if result['variant_pass'] else 'no'}`.",
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
    variants = {
        variant: _variant_analysis(variant)
        for variant in protocol.VARIANTS
    }
    passing = [
        variant for variant, result in variants.items()
        if result["variant_pass"]
    ]
    selected_variant = (
        max(
            passing,
            key=lambda variant: variants[variant][
                "control_validation_selected_score"],
        ) if passing else None
    )
    if selected_variant is not None:
        recommendation = (
            f"Freeze {selected_variant} with its control-validation-selected "
            "student and register a new untouched five-event confirmation. "
            "Do not reuse the sealed 100019-100151 confirmation events or "
            "reselect from this development audit."
        )
    else:
        return_pass = variants["mode_heads_return"]["student_seed_passes"]
        mode_pass = variants["mode_heads"]["student_seed_passes"]
        wide_pass = variants["wide_dagger"]["student_seed_passes"]
        recommendation = (
            "No compression variant passes. Compare the diagnostic pattern: "
            f"wide={wide_pass}/3, mode-head={mode_pass}/3, "
            f"return-aware={return_pass}/3. If mode heads improve over wide, "
            "conditional interference remains; if return-aware improves over "
            "plain mode heads, the remaining defect is closed-loop coverage. "
            "Otherwise retire single-network behavior cloning and train the "
            "conditioned policy directly against environment return."
        )
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "development_only": True,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "sealed_confirmation_event_seeds_not_used": list(
            protocol.SEALED_CONFIRMATION_EVENT_SEEDS),
        "thresholds": {
            "min_event_wins_vs_population": (
                protocol.MIN_EVENT_WINS_VS_POPULATION),
            "min_event_wins_vs_fixed_robust": (
                protocol.MIN_EVENT_WINS_VS_FIXED_ROBUST),
            "min_student_seed_passes": protocol.MIN_STUDENT_SEED_PASSES,
            "min_headroom_recovery": protocol.MIN_HEADROOM_RECOVERY,
            "max_mean_student_teacher_gap": (
                protocol.MAX_MEAN_STUDENT_TEACHER_GAP),
            "max_terminated_rate": protocol.MAX_TERMINATED_RATE,
        },
        "variants": variants,
        "promotion": selected_variant is not None,
        "selected_variant": selected_variant,
        "recommendation": recommendation,
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(f"CONTROL COMPRESSION ANALYSIS COMPLETE: {protocol.ANALYSIS_ROOT}")
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
