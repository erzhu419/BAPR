"""Analyze the frozen sticky-specialist routing screen."""
from __future__ import annotations

import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_sticky_router_v3 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_sticky_router_audit_v3 import (
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
    arms = {}
    for arm in protocol.ARMS:
        rows = [event["switching"][arm] for event in events]
        arms[arm] = {
            "mean": _mean(row["return_mean"] for row in rows),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(protocol.EVENT_SEEDS, rows)
            },
            "terminated_rate": _mean(row["terminated_rate"] for row in rows),
            "fallback_action_fraction": _mean(
                row["fallback_action_fraction"] for row in rows),
            "adaptive_mode_accuracy": _mean(
                row["adaptive_mode_accuracy"]
                for row in rows
                if row["adaptive_mode_accuracy"] is not None
            ) if any(
                row["adaptive_mode_accuracy"] is not None for row in rows
            ) else None,
        }
        posterior_rows = [row for row in rows if "posterior_metrics" in row]
        if posterior_rows:
            arms[arm]["posterior_mode_accuracy"] = _mean(
                row["posterior_metrics"]["mode_accuracy"]
                for row in posterior_rows
            )
        option_rows = [
            row for row in rows if "option_switch_count_mean" in row]
        if option_rows:
            arms[arm]["option_switch_count_mean"] = _mean(
                row["option_switch_count_mean"] for row in option_rows)

    robust = arms["robust_sac"]
    oracle = arms["dynamic_oracle"]
    for arm, row in arms.items():
        row["oracle_recovery"] = (
            (row["mean"] - robust["mean"])
            / (oracle["mean"] - robust["mean"])
            if oracle["mean"] > robust["mean"] else float("-inf")
        )
        row["relative_gain"] = (
            (row["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        row["event_wins_vs_robust"] = sum(
            row["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.EVENT_SEEDS
        )
    return {"arms": arms}


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }

    def both(arm: str, predicate) -> bool:
        return all(
            predicate(seeds[str(seed)]["arms"][arm])
            for seed in protocol.TRAINING_SEEDS
        )

    primary_pass = both(
        protocol.PRIMARY_ARM,
        lambda row: (
            row["relative_gain"] >= protocol.MIN_GAIN
            and row["oracle_recovery"] >= protocol.MIN_ORACLE_RECOVERY
            and row["mean"] >= protocol.MIN_SWITCHING_RETURN
            and row["terminated_rate"] == 0.0
            and row["event_wins_vs_robust"] == len(protocol.EVENT_SEEDS)
        ),
    )
    no_gate_useful = both(
        "posterior_map_no_gate",
        lambda row: row["relative_gain"] >= protocol.MIN_GAIN,
    )
    if primary_pass:
        diagnosis = "persistent atomic option switching repairs router thrashing"
        next_step = "freeze confirm3 and train fresh source banks for confirmation"
    elif no_gate_useful:
        diagnosis = (
            "mode information is useful, but frozen estimator trajectories remain "
            "insufficient for the registered sticky router"
        )
        next_step = (
            "fit the estimator on specialist-policy trajectories without changing "
            "the frozen confirm3 option"
        )
    else:
        diagnosis = "frozen estimator routing does not recover specialist headroom"
        next_step = "train switch-matched specialists and estimator jointly"
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "primary_arm": protocol.PRIMARY_ARM,
        "training_seed_metrics": seeds,
        "primary_pass": primary_pass,
        "no_gate_useful": no_gate_useful,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def report(payload: dict) -> str:
    lines = [
        "# Sticky independent-specialist router screen",
        "",
        "The primary arm confirms a new posterior mode for three consecutive "
        "transitions and then switches specialists atomically. It uses the robust "
        "controller only before the first option is identified.",
        "",
    ]
    for seed in protocol.TRAINING_SEEDS:
        rows = payload["training_seed_metrics"][str(seed)]["arms"]
        lines += [
            f"## Source seed {seed}",
            "",
            "| Arm | Return | Gain | Oracle recovery | Robust actions | Mode accuracy | Event wins |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in protocol.ARMS:
            row = rows[arm]
            accuracy = row.get("adaptive_mode_accuracy")
            accuracy_text = "-" if accuracy is None else f"{100*accuracy:.2f}%"
            lines.append(
                f"| `{arm}` | {row['mean']:.1f} | "
                f"{100*row['relative_gain']:.1f}% | "
                f"{100*row['oracle_recovery']:.1f}% | "
                f"{100*row['fallback_action_fraction']:.2f}% | "
                f"{accuracy_text} | "
                f"{row['event_wins_vs_robust']}/{len(protocol.EVENT_SEEDS)} |"
            )
        lines.append("")
    lines += [
        f"Primary confirm3 pass: **{payload['primary_pass']}**",
        "",
        f"Ungated MAP useful on both source seeds: **{payload['no_gate_useful']}**",
        "",
        f"Diagnosis: {payload['diagnosis']}.",
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
        "STICKY ROUTER ANALYSIS COMPLETE: "
        f"primary={payload['primary_pass']} "
        f"no_gate={payload['no_gate_useful']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
