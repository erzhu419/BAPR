"""Aggregate the checkpoint-only polarity policy-ensemble diagnostic."""
from __future__ import annotations

import math
import statistics
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_ensemble as protocol,
)
from jax_experiments.analysis.run_regime_polarity_policy_ensemble_audit import (
    validate_audit,
)


MIN_RELATIVE_GAIN = 0.10
MIN_HEADROOM_RECOVERY = 0.70


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _sd(values) -> float:
    values = [float(value) for value in values]
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _relative(delta: float, reference: float) -> float:
    return float(delta) / max(abs(float(reference)), 100.0)


def _event_rows(group: str) -> dict[int, dict[str, Any]]:
    rows = {}
    for event_seed in protocol.EVENT_SEEDS:
        validate_audit(group, event_seed)
        result = protocol.read_json(protocol.event_result(group, event_seed))
        switching = {
            row["arm"]: row for row in result["switching"]
        }
        stationary = {
            (row["arm"], int(row["mode"])): row
            for row in result["stationary"]
        }
        if (set(switching) != set(protocol.arm_labels(group))
                or set(stationary) != {
                    (arm, mode)
                    for arm in protocol.arm_labels(group)
                    for mode in protocol.MODES
                }):
            raise ValueError("policy-ensemble result matrix is incomplete")
        rows[event_seed] = {
            "switching": switching,
            "stationary": stationary,
        }
    return rows


def _arm_summary(events, arm: str) -> dict[str, Any]:
    switching = [
        value
        for event in events.values()
        for value in event["switching"][arm]["returns"]
    ]
    mode_returns = {
        str(mode): [
            value
            for event in events.values()
            for value in event["stationary"][(arm, mode)]["returns"]
        ]
        for mode in protocol.MODES
    }
    output = {
        "switching_mean": _mean(switching),
        "switching_sd": _sd(switching),
        "switching_termination": _mean(
            event["switching"][arm]["terminated_rate"]
            for event in events.values()),
        "action_disagreement_mean": _mean(
            event["switching"][arm]["action_disagreement_mean"]
            for event in events.values()),
        "stationary_by_mode": {
            mode: _mean(values) for mode, values in mode_returns.items()
        },
        "stationary_worst_mode": min(
            _mean(values) for values in mode_returns.values()),
    }
    posterior_rows = [
        event["switching"][arm].get("posterior_metrics")
        for event in events.values()
    ]
    if all(row is not None for row in posterior_rows):
        numeric_keys = sorted(set.intersection(*(
            {
                key for key, value in row.items()
                if isinstance(value, (int, float))
                and math.isfinite(float(value))
            }
            for row in posterior_rows
        )))
        output["posterior_metrics"] = {
            key: _mean(row[key] for row in posterior_rows)
            for key in numeric_keys
        }
    return output


def _comparison(events, candidate: str, reference: str) -> dict[str, Any]:
    event_deltas = {}
    paired_deltas = []
    references = []
    for event_seed, event in events.items():
        candidate_values = event["switching"][candidate]["returns"]
        reference_values = event["switching"][reference]["returns"]
        if len(candidate_values) != len(reference_values):
            raise ValueError("unpaired policy-ensemble switching vectors")
        deltas = [
            float(left) - float(right)
            for left, right in zip(candidate_values, reference_values)
        ]
        event_deltas[str(event_seed)] = _mean(deltas)
        paired_deltas.extend(deltas)
        references.extend(float(value) for value in reference_values)
    mean_delta = _mean(paired_deltas)
    reference_mean = _mean(references)
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_mean": reference_mean + mean_delta,
        "reference_mean": reference_mean,
        "mean_delta": mean_delta,
        "relative_gain": _relative(mean_delta, reference_mean),
        "event_seed_deltas": event_deltas,
        "event_seed_wins": sum(value > 0.0 for value in event_deltas.values()),
        "episode_wins": sum(value > 0.0 for value in paired_deltas),
        "n_episodes": len(paired_deltas),
    }


def _group_analysis(group: str) -> dict[str, Any]:
    events = _event_rows(group)
    summaries = {
        arm: _arm_summary(events, arm)
        for arm in protocol.arm_labels(group)
    }
    comparisons = {}
    for seed in protocol.controller_seeds(group):
        label = f"oracle_seed_{seed}_vs_robust_seed_{seed}"
        comparisons[label] = _comparison(
            events, f"oracle_seed_{seed}", f"robust_seed_{seed}")
    for reduction in ("mean", "median"):
        comparisons[f"oracle_{reduction}_vs_robust_{reduction}"] = (
            _comparison(
                events,
                f"oracle_{reduction}",
                f"robust_{reduction}",
            )
        )
        comparisons[f"learned_{reduction}_vs_robust_{reduction}"] = (
            _comparison(
                events,
                f"learned_{reduction}",
                f"robust_{reduction}",
            )
        )

    individual_robust_scores = [
        summaries[f"robust_seed_{seed}"]["switching_mean"]
        for seed in protocol.controller_seeds(group)
    ]
    individual_robust_baseline = {
        "mean": _mean(individual_robust_scores),
        "best": max(individual_robust_scores),
        "worst": min(individual_robust_scores),
    }
    ensemble_checks = {}
    for reduction in ("mean", "median"):
        oracle = comparisons[
            f"oracle_{reduction}_vs_robust_{reduction}"]
        learned = comparisons[
            f"learned_{reduction}_vs_robust_{reduction}"]
        headroom = float(oracle["mean_delta"])
        recovery = (
            float(learned["mean_delta"]) / headroom
            if headroom > 0.0 else None
        )
        oracle_pass = (
            float(oracle["relative_gain"]) >= MIN_RELATIVE_GAIN
            and int(oracle["event_seed_wins"]) == len(protocol.EVENT_SEEDS)
        )
        learned_pass = (
            oracle_pass
            and float(learned["relative_gain"]) >= MIN_RELATIVE_GAIN
            and int(learned["event_seed_wins"]) == len(protocol.EVENT_SEEDS)
            and recovery is not None
            and math.isfinite(recovery)
            and recovery >= MIN_HEADROOM_RECOVERY
        )
        ensemble_checks[reduction] = {
            "oracle_pass": oracle_pass,
            "learned_pass": learned_pass,
            "headroom_recovery": recovery,
            # The preregistered matched robust action ensemble is retained
            # above, but independently trained robust policies can average to
            # an invalid action.  These post-hoc diagnostics keep that collapse
            # from being mistaken for adaptation headroom.
            "oracle_delta_vs_individual_robust_mean": (
                summaries[f"oracle_{reduction}"]["switching_mean"]
                - individual_robust_baseline["mean"]),
            "learned_delta_vs_individual_robust_mean": (
                summaries[f"learned_{reduction}"]["switching_mean"]
                - individual_robust_baseline["mean"]),
            "learned_delta_vs_best_individual_robust": (
                summaries[f"learned_{reduction}"]["switching_mean"]
                - individual_robust_baseline["best"]),
        }
    return {
        "controller_seeds": list(protocol.controller_seeds(group)),
        "arm_summaries": summaries,
        "comparisons": comparisons,
        "ensemble_checks": ensemble_checks,
        "individual_robust_baseline": individual_robust_baseline,
        "individual_oracle_wins": sum(
            comparisons[
                f"oracle_seed_{seed}_vs_robust_seed_{seed}"
            ]["mean_delta"] > 0.0
            for seed in protocol.controller_seeds(group)
        ),
    }


def _recommendation(groups: dict[str, Any]) -> str:
    successful = [
        reduction
        for reduction in ("mean", "median")
        if all(
            groups[group]["ensemble_checks"][reduction]["learned_pass"]
            for group in protocol.GROUPS
        )
    ]
    if successful:
        selected = max(
            successful,
            key=lambda reduction: (
                min(
                    groups[group]["arm_summaries"][
                        f"learned_{reduction}"]["switching_mean"]
                    for group in protocol.GROUPS
                ),
                _mean(
                    groups[group]["arm_summaries"][
                        f"learned_{reduction}"]["switching_mean"]
                    for group in protocol.GROUPS
                ),
            ),
        )
        return (
            "Both action reductions pass the preregistered aggregate gate, "
            "but the matched robust action ensembles collapse and are not a "
            "credible strong baseline. Freeze the stronger diagnostic teacher "
            f"({selected}), distill one posterior-conditioned student, and "
            "compare that student against the individual robust-controller "
            "distribution on new event seeds."
        )
    oracle_successful = [
        reduction
        for reduction in ("mean", "median")
        if all(
            groups[group]["ensemble_checks"][reduction]["oracle_pass"]
            for group in protocol.GROUPS
        )
    ]
    if oracle_successful:
        return (
            "True-context aggregation is stable but the learned-context "
            "ensemble is not. Diagnose posterior-conditioned action geometry "
            "and train with soft/delayed beliefs; do not restart residual RL."
        )
    return (
        "Naive mean/median action aggregation does not stabilize the oracle "
        "controllers. Do not distill these aggregate actions. The next "
        "controller intervention must train a paired conditioned policy with "
        "explicit cross-seed stability, rather than another residual or gate."
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Polarity policy-ensemble diagnostic",
        "",
        "This is a retrospective, checkpoint-only controller-variance "
        "diagnostic. It is not a new confirmation experiment.",
        "",
    ]
    for group in protocol.GROUPS:
        result = payload["groups"][group]
        lines.extend([
            f"## {group.title()} controller group",
            "",
            "| arm | switching mean | delta vs matched robust | relative | "
            "event wins | action disagreement |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for reduction in ("mean", "median"):
            for kind in ("oracle", "learned"):
                label = f"{kind}_{reduction}"
                comparison = result["comparisons"][
                    f"{label}_vs_robust_{reduction}"]
                summary = result["arm_summaries"][label]
                lines.append(
                    f"| `{label}` | {summary['switching_mean']:.1f} | "
                    f"{comparison['mean_delta']:+.1f} | "
                    f"{100.0 * comparison['relative_gain']:+.1f}% | "
                    f"{comparison['event_seed_wins']}/3 | "
                    f"{summary['action_disagreement_mean']:.3f} |"
                )
        lines.extend([
            "",
            "| training seed | robust | oracle | delta |",
            "|---:|---:|---:|---:|",
        ])
        for seed in protocol.controller_seeds(group):
            comparison = result["comparisons"][
                f"oracle_seed_{seed}_vs_robust_seed_{seed}"]
            lines.append(
                f"| {seed} | {comparison['reference_mean']:.1f} | "
                f"{comparison['candidate_mean']:.1f} | "
                f"{comparison['mean_delta']:+.1f} |"
            )
        lines.extend([
            "",
            f"Individual oracle wins: `{result['individual_oracle_wins']}/5`.",
            "",
            "Individual robust switching baseline: "
            f"mean `{result['individual_robust_baseline']['mean']:.1f}`, "
            f"best `{result['individual_robust_baseline']['best']:.1f}`, "
            f"worst `{result['individual_robust_baseline']['worst']:.1f}`.",
            "",
        ])
    lines.extend([
        "## Decision",
        "",
        payload["recommendation"],
        "",
        "No GPU training or untouched-seed confirmation is launched by this "
        "diagnostic.",
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    protocol.validate_frozen_estimator()
    groups = {
        group: _group_analysis(group) for group in protocol.GROUPS
    }
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "development_only": True,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "event_seeds": list(protocol.EVENT_SEEDS),
        "thresholds": {
            "min_relative_gain": MIN_RELATIVE_GAIN,
            "min_headroom_recovery": MIN_HEADROOM_RECOVERY,
            "required_event_seed_wins": len(protocol.EVENT_SEEDS),
        },
        "groups": groups,
        "recommendation": _recommendation(groups),
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(f"POLICY ENSEMBLE ANALYSIS COMPLETE: {protocol.ANALYSIS_ROOT}")
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
