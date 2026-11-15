"""Analyze the causal independent-specialist router screen."""
from __future__ import annotations

import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_router_v1 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_router_audit_v1 import (
    validate_audit,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    events = [
        protocol.read_json(protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    ]
    stationary = {}
    switching = {}
    for arm in protocol.ARMS:
        by_mode = {
            str(mode): _mean(
                event["stationary"][arm][str(mode)]["return_mean"]
                for event in events
            )
            for mode in protocol.MODES
        }
        stationary[arm] = {
            "by_mode": by_mode,
            "mean": _mean(by_mode.values()),
            "worst": min(by_mode.values()),
            "terminated_rate": _mean(
                event["stationary"][arm][str(mode)]["terminated_rate"]
                for event in events
                for mode in protocol.MODES
            ),
        }
        switching[arm] = {
            "event_returns": {
                str(event_seed): float(event["switching"][arm]["return_mean"])
                for event_seed, event in zip(protocol.EVENT_SEEDS, events)
            },
            "mean": _mean(
                event["switching"][arm]["return_mean"] for event in events
            ),
            "terminated_rate": _mean(
                event["switching"][arm]["terminated_rate"] for event in events
            ),
        }
        if arm in protocol.LEARNED_ARMS:
            switching[arm].update({
                "fallback_action_fraction": _mean(
                    event["switching"][arm]["fallback_action_fraction"]
                    for event in events
                ),
                "posterior_mode_accuracy": _mean(
                    event["switching"][arm]["posterior_metrics"]["mode_accuracy"]
                    for event in events
                ),
                "adaptive_mode_accuracy": _mean(
                    event["switching"][arm]["adaptive_mode_accuracy"]
                    for event in events
                ),
            })

    robust_switch = switching["robust_sac"]["mean"]
    oracle_switch = switching["dynamic_oracle"]["mean"]
    robust_stationary = stationary["robust_sac"]["mean"]
    oracle_stationary = stationary["dynamic_oracle"]["mean"]
    candidates = {}
    for arm in protocol.LEARNED_ARMS:
        learned_switch = switching[arm]["mean"]
        learned_stationary = stationary[arm]["mean"]
        switching_recovery = (
            (learned_switch - robust_switch) / (oracle_switch - robust_switch)
            if oracle_switch > robust_switch else float("-inf")
        )
        stationary_recovery = (
            (learned_stationary - robust_stationary)
            / (oracle_stationary - robust_stationary)
            if oracle_stationary > robust_stationary else float("-inf")
        )
        event_wins = sum(
            switching[arm]["event_returns"][str(event_seed)]
            > switching["robust_sac"]["event_returns"][str(event_seed)]
            for event_seed in protocol.EVENT_SEEDS
        )
        checks = {
            "oracle_switching_gain_at_least_10pct": (
                oracle_switch - robust_switch
            ) / max(abs(robust_switch), 100.0) >= protocol.MIN_GAIN,
            "learned_switching_gain_at_least_10pct": (
                learned_switch - robust_switch
            ) / max(abs(robust_switch), 100.0) >= protocol.MIN_GAIN,
            "learned_switching_return_at_least_2200": (
                learned_switch >= protocol.MIN_SWITCHING_RETURN
            ),
            "switching_oracle_recovery_at_least_70pct": (
                switching_recovery >= protocol.MIN_ORACLE_RECOVERY
            ),
            "stationary_oracle_recovery_at_least_70pct": (
                stationary_recovery >= protocol.MIN_ORACLE_RECOVERY
            ),
            "learned_wins_all_event_streams": (
                event_wins == len(protocol.EVENT_SEEDS)
            ),
            "no_switching_termination_penalty": (
                switching[arm]["terminated_rate"]
                <= switching["robust_sac"]["terminated_rate"]
            ),
        }
        candidates[arm] = {
            "switching_oracle_recovery": switching_recovery,
            "stationary_oracle_recovery": stationary_recovery,
            "event_wins": event_wins,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {
        "stationary": stationary,
        "switching": switching,
        "candidates": candidates,
    }


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }
    passing_arms = [
        arm
        for arm in protocol.LEARNED_ARMS
        if all(
            seeds[str(seed)]["candidates"][arm]["pass"]
            for seed in protocol.TRAINING_SEEDS
        )
    ]
    selected = None
    if passing_arms:
        selected = max(
            passing_arms,
            key=lambda arm: min(
                seeds[str(seed)]["candidates"][arm][
                    "switching_oracle_recovery"
                ]
                for seed in protocol.TRAINING_SEEDS
            ),
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "training_seed_metrics": seeds,
        "passing_arms": passing_arms,
        "selected_arm": selected,
        "router_gate_pass": selected is not None,
        "next_step": (
            "train a fresh independent specialist bank and confirm the selected router"
            if selected is not None
            else "do not spend GPU budget on this specialist-router construction"
        ),
    }


def report(payload: dict) -> str:
    lines = [
        "# Causal independent-specialist router development screen",
        "",
        "This checkpoint-only screen reuses the two independent specialist "
        "banks, the frozen expected-action estimator, and the frozen causal "
        "fallback. True mode is available only to `dynamic_oracle`.",
        "",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["training_seed_metrics"][str(seed)]
        lines += [
            f"## Source seed {seed}",
            "",
            "| Arm | Stationary | Worst mode | Switching | Fallback |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in protocol.ARMS:
            fallback = row["switching"][arm].get("fallback_action_fraction")
            fallback_text = "-" if fallback is None else f"{100*fallback:.2f}%"
            lines.append(
                f"| `{arm}` | {row['stationary'][arm]['mean']:.1f} | "
                f"{row['stationary'][arm]['worst']:.1f} | "
                f"{row['switching'][arm]['mean']:.1f} | {fallback_text} |"
            )
        lines.append("")
        for arm in protocol.LEARNED_ARMS:
            candidate = row["candidates"][arm]
            lines.append(
                f"- `{arm}`: switching recovery "
                f"{100*candidate['switching_oracle_recovery']:.1f}%, "
                f"stationary recovery "
                f"{100*candidate['stationary_oracle_recovery']:.1f}%, "
                f"event wins {candidate['event_wins']}/"
                f"{len(protocol.EVENT_SEEDS)}, pass={candidate['pass']}."
            )
        lines.append("")
    lines += [
        f"Router gate pass: **{payload['router_gate_pass']}**",
        "",
        f"Selected development arm: `{payload['selected_arm']}`.",
        "",
        f"Decision: {payload['next_step']}.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "SPECIALIST ROUTER ANALYSIS COMPLETE: "
        f"pass={payload['router_gate_pass']} "
        f"selected={payload['selected_arm']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
